import json
import math


def loads(text: str):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number out of range")
        return number

    def bounded_int(value):
        number = int(value)
        if value == "-0" or not -(1 << 63) <= number < 1 << 64:
            return finite_float(value)
        return number

    def validate_strings(value, depth=0):
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, (list, dict)):
            if depth >= 128:
                raise ValueError("JSON recursion limit exceeded")
            for item in value:
                validate_strings(item, depth + 1)
                if isinstance(value, dict):
                    validate_strings(value[item], depth + 1)

    result = json.loads(
        text,
        parse_constant=invalid_constant,
        parse_float=finite_float,
        parse_int=bounded_int,
    )
    validate_strings(result)
    return result


def dumps(value, *, legacy_floats: bool = True) -> str:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return (
            "{"
            + ",".join(
                dumps(key) + ":" + dumps(value[key], legacy_floats=legacy_floats)
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return (
            "["
            + ",".join(dumps(item, legacy_floats=legacy_floats) for item in value)
            + "]"
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        result = repr(value)
        if "e" in result:
            mantissa, exponent_text = result.split("e")
            exponent = int(exponent_text)
            if -5 <= exponent < 16:
                negative = mantissa.startswith("-")
                digits = mantissa.lstrip("-").replace(".", "")
                decimal = exponent + 1
                if decimal <= 0:
                    result = "0." + "0" * -decimal + digits
                elif decimal >= len(digits):
                    result = digits + "0" * (decimal - len(digits)) + ".0"
                else:
                    result = digits[:decimal] + "." + digits[decimal:]
                return ("-" if negative else "") + result
            sign = "+" if exponent >= 0 and not legacy_floats else ""
            return mantissa + "e" + sign + str(exponent)
        return result
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
