import random
# import os
from scorer import Scorer

# random.seed(int(os.environ["SEED"]))

class YouBlewIt(object):
	def __init__(self, strategy, num_turns=None, stop_score=None):
		self.strategy = strategy
		self.num_turns = num_turns
		self.stop_score = stop_score
		self.total_score = 0

	def play(self):
		if self.num_turns:
			self._play_turns()
		else:
			self.num_turns = self._play_score()
		return (self.total_score, self.num_turns)

	def _play_score(self):
		turn_num = 0
		while (self.total_score < self.stop_score):
			score, actions = self._play_turn(turn_num)
			self.total_score = self.total_score + score
			turn_num = turn_num + 1
			# if score > 3500:
			# print score, actions
		return turn_num

	def _play_turns(self):
		for turn_num in x:
			_, actions = self._play_turn(turn_num)
			self.total_score = self.total_score + score

	def _play_turn(self, turn_num, current_score=0):
		num_remaining_dice = 6
		turn_actions = []
		new_score = 0
		new_actions = []
		scorer = Scorer([])
		while(self._should_roll(turn_num, num_remaining_dice, current_score)):
			die_rolls = self._roll(num_remaining_dice)
			turn_actions.append(('rolled', die_rolls))
			scorer = Scorer(die_rolls)
			if scorer.is_blown():
				return 0, turn_actions + ['blew it']
			actions = self.strategy.actions(die_rolls)
			turn_actions += actions
			score = scorer.apply_actions(actions)
			current_score = current_score + score
			turn_actions.append(('adding', score, current_score))
			num_remaining_dice = scorer.num_remaining_dice()
			num_remaining_dice = num_remaining_dice if not num_remaining_dice == 0 else 6
		dice = scorer._make_remaining_dice()
		num_remaining, raw_score = Scorer(dice).raw_score()
		current_score += raw_score
		turn_actions.append(('auto-adding', raw_score, current_score))
		game_over = self.stop_score and current_score + self.total_score >= self.stop_score
		if num_remaining == 0 and not game_over:
			turn_actions.append('rolled over')
			current_score, new_actions = self._play_turn(turn_num, current_score)
		return (current_score, turn_actions + new_actions)

	def _should_roll(self, turn_num, remaining_dice, current_score):
		has_to = self.total_score == 0 and current_score < 800
		game_over = self.stop_score and current_score + self.total_score >= self.stop_score
		strategy_says = self.strategy.should_roll(turn_num,
										     self.total_score,
										     remaining_dice,
										     current_score)
		if has_to:
			return True
		if game_over:
			return False
		return strategy_says

	def _roll(self, remaining_dice):
		return map(lambda _: random.randint(1, 6), xrange(remaining_dice))
