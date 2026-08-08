"""Small schema upgrades that preserve existing PostgreSQL volumes."""

from sqlalchemy import inspect, text


SENSOR_COLUMNS = {
    "humidity": "FLOAT",
    "accel_x": "FLOAT",
    "accel_y": "FLOAT",
    "accel_z": "FLOAT",
}


def migrate_sensor_columns(engine) -> None:
    """Add new nullable sensor columns when upgrading an existing database."""
    inspector = inspect(engine)
    if "motor_sensor_data" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("motor_sensor_data")}
    missing = [
        (name, sql_type)
        for name, sql_type in SENSOR_COLUMNS.items()
        if name not in existing
    ]
    if not missing:
        return

    with engine.begin() as connection:
        for name, sql_type in missing:
            # Names and types come only from the constant mapping above.
            connection.execute(
                text(f"ALTER TABLE motor_sensor_data ADD COLUMN {name} {sql_type}")
            )
