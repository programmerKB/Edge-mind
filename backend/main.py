import os
import json
import asyncio
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

from database import engine, Base, SessionLocal
from models import MotorSensorData
from tools import get_motor_status

load_dotenv()

# 初始化資料庫表格
Base.metadata.create_all(bind=engine)

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
            MotorSensorData(motor_id="M1", temperature=88.5, vibration=1.5, status="warning"),
            MotorSensorData(motor_id="M2", temperature=42.0, vibration=0.2, status="normal")
        ]
        db.add_all(test_records)
        db.commit()
    db.close()
# ---------------------------------------------

class ChatRequest(BaseModel):
    message: str

# ==============================================================================
# 【原本的路由】/api/chat (預設 ASCII 編碼，適合傳輸，終端機會顯示 \uXXXX)
# ==============================================================================
@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    async def agent_generator():
        yield f"data: {json.dumps({'status': 'thought', 'content': 'Agent 正在分析您的請求...'})}\n\n"
        await asyncio.sleep(0.5)

        config = types.GenerateContentConfig(
            system_instruction="你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，務必呼叫工具查詢最新狀態。回答請使用繁體中文，並給出具體的維護建議。",
            tools=[get_motor_status],
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
            system_instruction="你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，務必呼叫工具查詢最新狀態。回答請使用繁體中文，並給出具體的維護建議。",
            tools=[get_motor_status],
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