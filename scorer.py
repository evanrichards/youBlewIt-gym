class Scorer:
    values: list[int | None]
    combos: list[bool | None]

    def __init__(self, dice: list[int]) -> None:
        # skip	1		2		3		4		5
        self.values = [None, 0, 0, 0, 0, 0, 0]
        self.combos = [None, False, False, False, False, False, False]
        for value in dice:
            current = self.values[value]
            assert current is not None
            self.values[value] = current + 1
        for num, value in enumerate(self.values):
            if num == 0:
                continue
            if value is not None and value >= 3:
                self.combos[num] = True

    def has_thousand_combo(self) -> bool:
        return bool(self.combos[1])

    def has_two_hundred_combo(self) -> bool:
        return bool(self.combos[2])

    def has_three_hundred_combo(self) -> bool:
        return bool(self.combos[3])

    def has_four_hundred_combo(self) -> bool:
        return bool(self.combos[4])

    def has_five_hundred_combo(self) -> bool:
        return bool(self.combos[5])

    def has_six_hundred_combo(self) -> bool:
        return bool(self.combos[6])

    def has_ones(self) -> bool:
        return self.values[1] != 0

    def has_fifties(self) -> bool:
        return self.values[5] != 0

    def take_thousand_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(1)

    def take_two_hundred_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(2)

    def take_three_hundred_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(3)

    def take_four_hundred_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(4)

    def take_five_hundred_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(5)

    def take_six_hundred_combo(self) -> tuple[int, list[int]]:
        return self._take_gen(6)

    def take_one(self) -> tuple[int, list[int]]:
        return self._take_gen(1, 1)

    def take_ones(self, num_ones: int) -> tuple[int, list[int]]:
        return self._take_gen(1, num_ones)

    def take_fifty(self) -> tuple[int, list[int]]:
        return self._take_gen(5, 1)

    def take_fifties(self, num_fifties: int) -> tuple[int, list[int]]:
        return self._take_gen(5, num_fifties)

    def _take_gen(self, loc: int, num_to_take: int = 3) -> tuple[int, list[int]]:
        value = self.values[loc]
        if value is None or value < num_to_take:
            raise AssertionError("Removed dice that were not available")
        self.values[loc] = value - num_to_take
        if num_to_take == 3:
            return loc * 100 if loc != 1 else 1000, self._make_remaining_dice()
        elif loc == 1:
            return 100 * num_to_take, self._make_remaining_dice()
        elif loc == 5:
            return 50 * num_to_take, self._make_remaining_dice()
        raise AssertionError(f"Invalid take_gen call: loc={loc}, num_to_take={num_to_take}")

    def _make_remaining_dice(self) -> list[int]:
        remaining: list[int] = []
        for loc, value in enumerate(self.values):
            if value:
                for _count in range(value):
                    remaining.append(loc)
        return remaining

    def num_ones(self) -> int:
        value = self.values[1]
        return value if value is not None else 0

    def num_fifties(self) -> int:
        value = self.values[5]
        return value if value is not None else 0

    def is_blown(self) -> bool:
        return not (
            self.has_two_hundred_combo()
            or self.has_three_hundred_combo()
            or self.has_four_hundred_combo()
            or self.has_six_hundred_combo()
            or self.has_ones()
            or self.has_fifties()
        )

    def apply_actions(self, actions: list[str]) -> int:
        total_score = 0
        for action in actions:
            method = getattr(self, action)
            score, _ = method()
            total_score += score
        return total_score

    def num_remaining_dice(self) -> int:
        return sum(v for v in self.values[1:] if v is not None)

    def raw_score(self) -> tuple[int, int]:
        score = 0
        remaining_dice: list[int] = []
        if self.is_blown():
            return len(self._make_remaining_dice()), 0
        if self.has_thousand_combo():
            add_score, remaining_dice = self.take_thousand_combo()
            score += add_score
        if self.has_six_hundred_combo():
            add_score, remaining_dice = self.take_six_hundred_combo()
            score += add_score
        if self.has_five_hundred_combo():
            add_score, remaining_dice = self.take_five_hundred_combo()
            score += add_score
        if self.has_four_hundred_combo():
            add_score, remaining_dice = self.take_four_hundred_combo()
            score += add_score
        if self.has_three_hundred_combo():
            add_score, remaining_dice = self.take_three_hundred_combo()
            score += add_score
        if self.has_ones():
            num_ones = self.num_ones()
            add_score, remaining_dice = self.take_ones(num_ones)
            score += add_score
        if self.has_two_hundred_combo():
            add_score, remaining_dice = self.take_two_hundred_combo()
            score += add_score
        if self.has_fifties():
            num_fifties = self.num_fifties()
            add_score, remaining_dice = self.take_fifties(num_fifties)
            score += add_score
        return len(remaining_dice), score
