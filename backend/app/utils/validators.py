"""Input range validation utilities."""


def validate_scan_rate(sr: float) -> bool:
    return 1.0 <= sr <= 200.0


def validate_voltage_range(v_min: float, v_max: float) -> bool:
    return v_min < v_max and -2.0 <= v_min and v_max <= 2.0
