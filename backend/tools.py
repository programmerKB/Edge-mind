import json
from database import SessionLocal
from models import MotorSensorData

def get_motor_status(motor_id: str) -> str:
    """
    查詢指定馬達/設備的最新感測器狀態（包含溫度、震動數據與狀態評估）。
    
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
            "vibration": record.vibration,
            "status": record.status,
            "recorded_at": record.recorded_at.isoformat() if record.recorded_at else None
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()