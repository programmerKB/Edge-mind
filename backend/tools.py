import json
from database import SessionLocal
from models import MotorSensorData
from forecasting import ForecastError, forecast_temperature

def get_motor_status(motor_id: str) -> str:
    """
    查詢指定馬達/設備的最新感測器狀態（包含溫度、濕度、三軸與狀態評估）。
    
    Args:
        motor_id: 馬達編號，例如 'M1', 'M2', 'STM32-Node-1'
    """
    db = SessionLocal()
    try:
        # 查詢該馬達最新的一筆紀錄
        record = db.query(MotorSensorData).filter(
            MotorSensorData.motor_id == motor_id
        ).order_by(MotorSensorData.recorded_at.desc()).first()
        
        if not record:
            return json.dumps({"error": f"資料庫中找不到馬達 {motor_id} 的感測器資料"})
            
        result = {
            "motor_id": record.motor_id,
            "temperature": record.temperature,
            "humidity": record.humidity,
            "accel_x": record.accel_x,
            "accel_y": record.accel_y,
            "accel_z": record.accel_z,
            "vibration": record.vibration,
            "status": record.status,
            "recorded_at": record.recorded_at.isoformat() if record.recorded_at else None
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()


def get_temperature_forecast(
    motor_id: str,
    training_motor_id: str | None = None,
) -> str:
    """預測指定馬達/設備 30 分鐘後的溫度。

    使用最新的溫度、濕度與三軸加速度 XYZ，並回傳模型驗證誤差。

    Args:
        motor_id: 推論資料的馬達編號，例如 'DEMO-2'、'M1'
        training_motor_id: 訓練模型的資料來源；未提供時使用 motor_id
    """
    db = SessionLocal()
    try:
        return json.dumps(
            forecast_temperature(
                db,
                motor_id,
                auto_train=True,
                training_motor_id=training_motor_id,
            ),
            ensure_ascii=False,
        )
    except ForecastError as error:
        db.rollback()
        return json.dumps({"error": str(error)}, ensure_ascii=False)
    except Exception as error:
        db.rollback()
        return json.dumps(
            {"error": f"預測失敗：{error}"},
            ensure_ascii=False,
        )
    finally:
        db.close()
