"""Small schema upgrades that preserve existing PostgreSQL volumes."""

from sqlalchemy import inspect, text


SENSOR_COLUMNS = {
    "humidity": "FLOAT",
    "accel_x": "FLOAT",
    "accel_y": "FLOAT",
    "accel_z": "FLOAT",
}
SENSOR_TIME_INDEX = "ix_motor_sensor_motor_recorded_at"


def migrate_sensor_columns(engine) -> None:
    """Add feature columns and the main history-query index when missing."""
    inspector = inspect(engine)
    if "motor_sensor_data" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("motor_sensor_data")}
    missing = [
        (name, sql_type)
        for name, sql_type in SENSOR_COLUMNS.items()
        if name not in existing
    ]
    with engine.begin() as connection:
        for name, sql_type in missing:
            # Names and types come only from the constant mapping above.
            connection.execute(
                text(f"ALTER TABLE motor_sensor_data ADD COLUMN {name} {sql_type}")
            )
        existing_indexes = {
            index["name"] for index in inspector.get_indexes("motor_sensor_data")
        }
        if SENSOR_TIME_INDEX not in existing_indexes:
            # Every status/forecast query filters by motor and orders by time.
            connection.execute(
                text(
                    "CREATE INDEX ix_motor_sensor_motor_recorded_at "
                    "ON motor_sensor_data (motor_id, recorded_at)"
                )
            )
