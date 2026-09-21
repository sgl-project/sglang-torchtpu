"""Generated from scripts/raiden_runtime_verification/formulas.ltl. Do not edit by hand."""

import logging

logger = logging.getLogger(__name__)

_ATOMIC_PROPOSITIONS: tuple[str, ...] = (
    "call_completed",
    "call_failed",
    "call_started",
    "on_call",
    "push_registered_plan",
    "register_active_plan",
    "register_read",
    "start_read",
    "unregister_active_plan",
    "clear_status",
    "kv_poll_bootstrapping",
    "kv_poll_failed",
    "kv_poll_no_entry",
    "kv_poll_success",
    "kv_poll_transferring",
    "kv_poll_waiting_for_input",
    "on_return",
    "raiden_receiver_poll",
    "raiden_sender_poll",
    "update_status",
)
_INITIAL_STATE: int = 0
_CONDITION_NODES: dict[int, tuple[int, int, int]] = {
    2: (0, 3, 36),
    3: (1, 4, 36),
    4: (2, 5, 36),
    5: (3, 6, 36),
    6: (4, 7, 36),
    7: (5, 8, 36),
    8: (6, 9, 36),
    9: (7, 10, 36),
    10: (8, 11, 36),
    11: (9, 12, 29),
    12: (10, 13, 25),
    13: (11, 14, 25),
    14: (12, 15, 26),
    15: (13, 16, 25),
    16: (14, 17, 25),
    17: (15, 18, 25),
    18: (16, 19, 22),
    19: (17, 20, 1),
    20: (18, 21, 1),
    21: (19, 0, 1),
    22: (17, 23, 0),
    23: (18, 24, 0),
    24: (19, 1, 0),
    25: (16, 1, 22),
    26: (13, 27, 25),
    27: (14, 28, 25),
    28: (15, 1, 25),
    29: (10, 30, 32),
    30: (11, 31, 32),
    31: (12, 32, 33),
    32: (16, 1, 0),
    33: (13, 34, 32),
    34: (14, 35, 32),
    35: (15, 1, 32),
    36: (9, 37, 29),
    37: (10, 38, 25),
    38: (11, 39, 25),
    39: (12, 25, 26),
    40: (9, 41, 48),
    41: (10, 42, 0),
    42: (11, 43, 0),
    43: (12, 44, 0),
    44: (13, 45, 0),
    45: (14, 46, 0),
    46: (15, 0, 47),
    47: (16, 0, 19),
    48: (10, 49, 0),
    49: (11, 50, 0),
    50: (12, 51, 0),
    51: (13, 52, 0),
    52: (14, 53, 0),
    53: (15, 0, 54),
    54: (16, 0, 1),
    55: (9, 56, 62),
    56: (10, 57, 0),
    57: (11, 58, 0),
    58: (12, 59, 0),
    59: (13, 60, 0),
    60: (14, 0, 61),
    61: (15, 47, 0),
    62: (10, 63, 0),
    63: (11, 64, 0),
    64: (12, 65, 0),
    65: (13, 66, 0),
    66: (14, 0, 67),
    67: (15, 54, 0),
    68: (9, 69, 74),
    69: (10, 70, 0),
    70: (11, 71, 0),
    71: (12, 72, 0),
    72: (13, 0, 73),
    73: (14, 61, 0),
    74: (10, 75, 0),
    75: (11, 76, 0),
    76: (12, 77, 0),
    77: (13, 0, 78),
    78: (14, 67, 0),
    79: (9, 80, 84),
    80: (10, 81, 0),
    81: (11, 0, 82),
    82: (12, 83, 0),
    83: (13, 73, 0),
    84: (10, 85, 0),
    85: (11, 0, 86),
    86: (12, 87, 0),
    87: (13, 78, 0),
    88: (9, 89, 91),
    89: (10, 0, 90),
    90: (11, 82, 0),
    91: (10, 0, 92),
    92: (11, 86, 0),
    93: (0, 94, 115),
    94: (1, 95, 115),
    95: (2, 96, 115),
    96: (3, 97, 115),
    97: (4, 98, 115),
    98: (5, 99, 115),
    99: (6, 100, 115),
    100: (7, 101, 115),
    101: (8, 102, 115),
    102: (9, 103, 109),
    103: (10, 104, 25),
    104: (11, 105, 25),
    105: (12, 106, 25),
    106: (13, 107, 25),
    107: (14, 108, 25),
    108: (15, 18, 1),
    109: (10, 110, 32),
    110: (11, 111, 32),
    111: (12, 112, 32),
    112: (13, 113, 32),
    113: (14, 114, 32),
    114: (15, 32, 1),
    115: (9, 116, 109),
    116: (10, 117, 25),
    117: (11, 118, 25),
    118: (12, 119, 25),
    119: (13, 120, 25),
    120: (14, 121, 25),
    121: (15, 25, 1),
    122: (0, 123, 142),
    123: (1, 124, 142),
    124: (2, 125, 142),
    125: (3, 126, 142),
    126: (4, 127, 142),
    127: (5, 128, 142),
    128: (6, 129, 142),
    129: (7, 130, 142),
    130: (8, 131, 142),
    131: (9, 132, 137),
    132: (10, 133, 25),
    133: (11, 134, 25),
    134: (12, 135, 25),
    135: (13, 136, 25),
    136: (14, 17, 28),
    137: (10, 138, 32),
    138: (11, 139, 32),
    139: (12, 140, 32),
    140: (13, 141, 32),
    141: (14, 32, 35),
    142: (9, 143, 137),
    143: (10, 144, 25),
    144: (11, 145, 25),
    145: (12, 146, 25),
    146: (13, 147, 25),
    147: (14, 25, 28),
    148: (9, 149, 152),
    149: (10, 150, 0),
    150: (11, 151, 0),
    151: (12, 0, 83),
    152: (10, 153, 0),
    153: (11, 154, 0),
    154: (12, 0, 87),
    155: (0, 156, 173),
    156: (1, 157, 173),
    157: (2, 158, 173),
    158: (3, 159, 173),
    159: (4, 160, 173),
    160: (5, 161, 173),
    161: (6, 162, 173),
    162: (7, 163, 173),
    163: (8, 164, 173),
    164: (9, 165, 169),
    165: (10, 166, 25),
    166: (11, 167, 25),
    167: (12, 168, 25),
    168: (13, 16, 27),
    169: (10, 170, 32),
    170: (11, 171, 32),
    171: (12, 172, 32),
    172: (13, 32, 34),
    173: (9, 174, 169),
    174: (10, 175, 25),
    175: (11, 176, 25),
    176: (12, 177, 25),
    177: (13, 25, 27),
    178: (0, 179, 195),
    179: (1, 180, 195),
    180: (2, 181, 195),
    181: (3, 182, 195),
    182: (4, 183, 195),
    183: (5, 184, 195),
    184: (6, 185, 195),
    185: (7, 186, 195),
    186: (8, 187, 195),
    187: (9, 188, 192),
    188: (10, 189, 25),
    189: (11, 190, 191),
    190: (12, 15, 25),
    191: (12, 26, 25),
    192: (10, 193, 32),
    193: (11, 32, 194),
    194: (12, 33, 32),
    195: (9, 196, 192),
    196: (10, 197, 25),
    197: (11, 25, 191),
    198: (0, 199, 213),
    199: (1, 200, 213),
    200: (2, 201, 213),
    201: (3, 202, 213),
    202: (4, 203, 213),
    203: (5, 204, 213),
    204: (6, 205, 213),
    205: (7, 206, 213),
    206: (8, 207, 213),
    207: (9, 208, 211),
    208: (10, 209, 210),
    209: (11, 190, 25),
    210: (11, 191, 25),
    211: (10, 32, 212),
    212: (11, 194, 32),
    213: (9, 214, 211),
    214: (10, 25, 210),
}
_TRANSITIONS: dict[int, tuple[tuple[int, int], ...]] = {
    0: (
        (2, 0),
        (40, 1),
        (55, 2),
        (68, 3),
        (79, 4),
        (88, 5),
    ),
    1: (
        (93, 1),
        (55, 2),
        (79, 4),
    ),
    2: (
        (122, 2),
        (68, 3),
        (79, 4),
    ),
    3: (
        (148, 0),
        (155, 3),
        (79, 4),
    ),
    4: (
        (148, 0),
        (178, 4),
    ),
    5: (
        (40, 1),
        (79, 4),
        (198, 5),
    ),
}


def _condition_holds(condition: int, valuation: int) -> bool:
    while condition > 1:
        proposition, low, high = _CONDITION_NODES[condition]
        condition = high if valuation & (1 << proposition) else low
    return condition == 1


class GeneratedRuntimeMonitor:
    def __init__(self, *, trace_id: object | None = None) -> None:
        self.trace_id = trace_id
        self.state = _INITIAL_STATE
        self.violated = False

    @property
    def atomic_propositions(self) -> tuple[str, ...]:
        return _ATOMIC_PROPOSITIONS

    def step(self, true_propositions: set[str]) -> bool:
        if self.violated:
            logger.info(
                "GeneratedRuntimeMonitor trace_id=%s state=%s "
                "atomic_propositions=%s accepted=False already_violated=True",
                self.trace_id,
                self.state,
                sorted(true_propositions),
            )
            return False

        unknown = true_propositions - set(_ATOMIC_PROPOSITIONS)
        if unknown:
            raise ValueError(f"unknown proposition(s): {', '.join(sorted(unknown))}")

        valuation = sum(
            1 << index
            for index, name in enumerate(_ATOMIC_PROPOSITIONS)
            if name in true_propositions
        )
        source = self.state
        for condition, destination in _TRANSITIONS[self.state]:
            if _condition_holds(condition, valuation):
                self.state = destination
                logger.info(
                    "GeneratedRuntimeMonitor trace_id=%s transition=%s->%s "
                    "atomic_propositions=%s accepted=True",
                    self.trace_id,
                    source,
                    destination,
                    sorted(true_propositions),
                )
                return True

        self.violated = True
        logger.error(
            "GeneratedRuntimeMonitor trace_id=%s state=%s "
            "atomic_propositions=%s accepted=False",
            self.trace_id,
            source,
            sorted(true_propositions),
        )
        return False
