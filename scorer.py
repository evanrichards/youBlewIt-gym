class Scorer(object):

	def __init__(self, dice):
		     		 # skip	1		2		3		4		5	
		self.values = [None, 0,		0,		0,		0,     0,     0]
		self.combos = [None, False,	False,	False,	False, False, False]
		for value in dice:
			self.values[value] = self.values[value] + 1
		for num, value in enumerate(self.values):
			if num == 0:
				continue
			if value >= 3:
				self.combos[num] = True

	def has_thousand_combo(self):
		return self.combos[1]

	def has_two_hundred_combo(self):
		return self.combos[2]

	def has_three_hundred_combo(self):
		return self.combos[3]

	def has_four_hundred_combo(self):
		return self.combos[4]

	def has_five_hundred_combo(self):
		return self.combos[5]

	def has_six_hundred_combo(self):
		return self.combos[6]

	def has_ones(self):
		return not self.values[1] == 0

	def has_fifties(self):
		return not self.values[5] == 0

	def take_thousand_combo(self):
		return self.take_gen(1)

	def take_two_hundred_combo(self):
		return self.take_gen(2)

	def take_three_hundred_combo(self):
		return self.take_gen(3)

	def take_four_hundred_combo(self):
		return self.take_gen(4)

	def take_five_hundred_combo(self):
		return self.take_gen(5)

	def take_six_hundred_combo(self):
		return self.take_gen(6)

	def take_one(self):
		return self.take_gen(1, 1)

	def take_ones(self, num_ones):
		return self.take_gen(1, num_ones)

	def take_fifty(self):
		return self.take_gen(5, 1)

	def take_fifties(self, num_fifties):
		return self.take_gen(5, num_fifties)

	def take_gen(self, loc, num_to_take=3):
		if self.values[loc] < num_to_take:
			assert False, "Removed dice that were not available"
		self.values[loc] -= num_to_take
		if num_to_take == 3:
			return loc * 100 if not loc == 1 else 1000, self._make_remaining_dice()
		elif loc == 1:
			return 100 * num_to_take, self._make_remaining_dice()
		elif loc == 5:
			return 50 * num_to_take, self._make_remaining_dice()


	def _make_remaining_dice(self):
		remaining = []
		count = 0
		for loc, value in enumerate(self.values):
			if value:
				for count in xrange(value):
					remaining.append(loc)
		return remaining

	def num_ones(self):
		return self.values[1]

	def num_fifties(self):
		return self.values[5]

	def is_blown(self):
		return not (self.has_two_hundred_combo()
				or self.has_three_hundred_combo()
				or self.has_four_hundred_combo()
				or self.has_six_hundred_combo()
				or self.has_ones()
				or self.has_fifties())

	def apply_actions(self, actions):
		total_score = 0
		for action in actions:
			score, a = self.__getattribute__(action)()
			total_score += score
		return total_score

	def num_remaining_dice(self):
		return sum(self.values[1:])

	def raw_score(self):
		# print "scoring", die_rolls
		score = 0
		if self.is_blown():
			return self._make_remaining_dice(), 0
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
		# print "for", add_score, "points"
		return len(remaining_dice), score
