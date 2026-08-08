import os
import json
import asyncio
import math
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

from database import engine, Base, SessionLocal, get_db
from models import MotorSensorData
from migrations import migrate_sensor_columns
from forecasting import ForecastError, forecast_temperature, train_and_save_model
from tools import get_motor_status, get_temperature_forecast

load_dotenv()

# 初始化資料庫表格
Base.metadata.create_all(bind=engine)
migrate_sensor_columns(engine)

app = FastAPI(title="邊緣設備診斷 Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()
# 已經確認使用最新的輕量化模型
MODEL_ID = "gemini-3.5-flash-lite"

# --- 啟動時自動注入測試資料 (僅供開發測試用) ---
@app.on_event("startup")
def populate_test_data():
    db = SessionLocal()
    if db.query(MotorSensorData).count() == 0:
        test_records = [
            MotorSensorData(
                motor_id="M1",
                temperature=88.5,
                humidity=68.0,
                accel_x=0.12,
                accel_y=0.08,
                accel_z=1.03,
                vibration=1.5,
                status="warning",
            ),
            MotorSensorData(
                motor_id="M2",
                temperature=42.0,
                humidity=51.0,
                accel_x=0.02,
                accel_y=0.01,
                accel_z=0.99,
                vibration=0.2,
                status="normal",
            ),
        ]
        db.add_all(test_records)
        db.commit()
    db.close()
# ---------------------------------------------

class ChatRequest(BaseModel):
    message: str

# ==============================================================================
class SensorReadingRequest(BaseModel):
    motor_id: str = Field(min_length=1, max_length=100)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    accel_x: float
    accel_y: float
    accel_z: float
    recorded_at: datetime | None = None
    status: str = Field(default="normal", max_length=50)


@app.post("/api/sensor-readings", status_code=status.HTTP_201_CREATED)
def create_sensor_reading(
    request: SensorReadingRequest,
    db=Depends(get_db),
):
    """Receive one complete feature vector from an edge sensor."""
    values = (
        request.temperature,
        request.humidity,
        request.accel_x,
        request.accel_y,
        request.accel_z,
    )
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="所有感測值都必須是有限數值",
        )

    record = MotorSensorData(
        motor_id=request.motor_id,
        temperature=request.temperature,
        humidity=request.humidity,
        accel_x=request.accel_x,
        accel_y=request.accel_y,
        accel_z=request.accel_z,
        vibration=math.sqrt(
            request.accel_x ** 2
            + request.accel_y ** 2
            + request.accel_z ** 2
        ),
        status=request.status,
    )
    if request.recorded_at is not None:
        record.recorded_at = request.recorded_at
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"感測資料寫入失敗：{error}",
        ) from error
    return {
        "id": record.id,
        "motor_id": record.motor_id,
        "temperature": record.temperature,
        "humidity": record.humidity,
        "accel_x": record.accel_x,
        "accel_y": record.accel_y,
        "accel_z": record.accel_z,
        "recorded_at": (
            record.recorded_at.isoformat()
            if record.recorded_at
            else None
        ),
    }


@app.post("/api/predictions/train/{motor_id}")
def train_temperature_forecast(
    motor_id: str,
    db=Depends(get_db),
):
    """Train and persist a per-device model using historical readings."""
    try:
        payload = train_and_save_model(db, motor_id)
        return {
            "motor_id": motor_id,
            "feature_names": payload["feature_names"],
            "forecast_horizon_minutes": payload["horizon_minutes"],
            "sample_count": payload["sample_count"],
            "validation_sample_count": payload["validation_sample_count"],
            "validation_mae": round(payload["mae"], 4),
            "validation_rmse": round(payload["rmse"], 4),
            "trained_at": payload["trained_at"],
        }
    except ForecastError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型訓練失敗：{error}",
        ) from error


@app.get("/api/predictions/temperature/{motor_id}")
def get_30_minute_temperature_forecast(
    motor_id: str,
    auto_train: bool = True,
    db=Depends(get_db),
):
    """Predict temperature 30 minutes after the latest complete reading."""
    try:
        return forecast_temperature(db, motor_id, auto_train=auto_train)
    except ForecastError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"溫度預測失敗：{error}",
        ) from error

# 【原本的路由】/api/chat (預設 ASCII 編碼，適合傳輸，終端機會顯示 \uXXXX)
# ==============================================================================
@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    async def agent_generator():
        yield f"data: {json.dumps({'status': 'thought', 'content': 'Agent 正在分析您的請求...'})}\n\n"
        await asyncio.sleep(0.5)

        config = types.GenerateContentConfig(
            system_instruction="你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，查詢狀態時呼叫狀態工具，詢問未來溫度時務必呼叫 30 分鐘預測工具。回答請使用繁體中文，並給出具體的維護建議。",
            tools=[get_motor_status, get_temperature_forecast],
            temperature=0.2
        )

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=request.message)])
        ]

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=config
        )

        if response.function_calls:
            contents.append(response.candidates[0].content)
            tool_response_parts = []
            
            for tool_call in response.function_calls:
                func_name = tool_call.name
                func_args = tool_call.args
                
                yield f"data: {json.dumps({'status': 'action', 'content': f'正在呼叫工具: {func_name}，參數: {func_args}'})}\n\n"
                
                if func_name == "get_motor_status":
                    motor_id = func_args.get("motor_id", "")
                    tool_result_str = get_motor_status(motor_id=motor_id)
                elif func_name == "get_temperature_forecast":
                    motor_id = func_args.get("motor_id", "")
                    tool_result_str = get_temperature_forecast(motor_id=motor_id)
                else:
                    tool_result_str = json.dumps({"error": "未知的工具"})
                
                yield f"data: {json.dumps({'status': 'observation', 'content': f'資料庫回傳: {tool_result_str}'})}\n\n"
                
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"result": json.loads(tool_result_str)}
                    )
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))
            yield f"data: {json.dumps({'status': 'thought', 'content': '正在統整邊緣感測數據並生成報告...'})}\n\n"

            final_response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config
            )
            
            yield f"data: {json.dumps({'status': 'success', 'content': final_response.text})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'success', 'content': response.text})}\n\n"

    return StreamingResponse(agent_generator(), media_type="text/event-stream")


# ==============================================================================
# 【新增的路由】/api/chat_utf8 (強制保留中文，終端機會直接顯示正常中文字)
# ==============================================================================
@app.post("/api/chat_utf8")
async def chat_with_agent_utf8(request: ChatRequest):
    async def agent_generator():
        # 加入 ensure_ascii=False 讓中文正常顯示
        yield f"data: {json.dumps({'status': 'thought', 'content': 'Agent 正在分析您的請求...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        config = types.GenerateContentConfig(
            system_instruction="你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，查詢狀態時呼叫狀態工具，詢問未來溫度時務必呼叫 30 分鐘預測工具。回答請使用繁體中文，並給出具體的維護建議。",
            tools=[get_motor_status, get_temperature_forecast],
            temperature=0.2
        )

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=request.message)])
        ]

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=config
        )

        if response.function_calls:
            contents.append(response.candidates[0].content)
            tool_response_parts = []
            
            for tool_call in response.function_calls:
                func_name = tool_call.name
                func_args = tool_call.args
                
                yield f"data: {json.dumps({'status': 'action', 'content': f'正在呼叫工具: {func_name}，參數: {func_args}'}, ensure_ascii=False)}\n\n"
                
                if func_name == "get_motor_status":
                    motor_id = func_args.get("motor_id", "")
                    tool_result_str = get_motor_status(motor_id=motor_id)
                elif func_name == "get_temperature_forecast":
                    motor_id = func_args.get("motor_id", "")
                    tool_result_str = get_temperature_forecast(motor_id=motor_id)
                else:
                    tool_result_str = json.dumps({"error": "未知的工具"})
                
                yield f"data: {json.dumps({'status': 'observation', 'content': f'資料庫回傳: {tool_result_str}'}, ensure_ascii=False)}\n\n"
                
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"result": json.loads(tool_result_str)}
                    )
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))
            yield f"data: {json.dumps({'status': 'thought', 'content': '正在統整邊緣感測數據並生成報告...'}, ensure_ascii=False)}\n\n"

            final_response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config
            )
            
            yield f"data: {json.dumps({'status': 'success', 'content': final_response.text}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'success', 'content': response.text}, ensure_ascii=False)}\n\n"

    return StreamingResponse(agent_generator(), media_type="text/event-stream")