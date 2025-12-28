"""PPO-based strategy using a trained MaskablePPO model."""

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from scorer import Scorer


def score_to_bucket(score: int, bucket_size: int = 2000, num_buckets: int = 5) -> int:
    """Convert score to bucket index (0-4 for 0-2k, 2k-4k, etc.)."""
    return min(score // bucket_size, num_buckets - 1)


class PPOStrategy:
    """Strategy that uses a trained MaskablePPO model to make decisions."""

    model: MaskablePPO

    # Internal game state
    dice: list[int]
    unbanked_score: int
    agent_score: int
    opponent_score: int
    just_rolled: bool

    # Action cache to avoid redundant predictions
    _cached_action: int | None
    _cached_dice: tuple[int, ...] | None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize PPO strategy with trained model.

        Args:
            model_path: Path to model zip file. If None, uses default models/best_1v1_model.zip
        """
        if model_path is None:
            # Default to models/best_1v1_model.zip in repo root
            repo_root = Path(__file__).parent.parent
            model_path = str(repo_root / "models" / "best_1v1_model.zip")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = MaskablePPO.load(model_path)

        # Initialize state
        self.dice = []
        self.unbanked_score = 0
        self.agent_score = 0
        self.opponent_score = 0
        self.just_rolled = False

        # Cache
        self._cached_action = None
        self._cached_dice = None

    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool:
        """Decide whether to roll or bank.

        Args:
            turn_num: Current turn number (unused)
            total_score: Player's total banked score
            remaining_dice: Number of dice available to roll
            current_score: Unbanked turn score

        Returns:
            True to roll, False to bank
        """
        # Update internal state
        self.agent_score = total_score
        self.unbanked_score = current_score

        # Build observation (we're deciding between roll and bank, so no dice on table)
        # This represents the state after taking dice or at start of turn
        obs = self._build_observation(
            dice_remaining=remaining_dice,
            must_roll=(remaining_dice == 6 and current_score == 0),  # Start of turn
            unbanked_score=current_score,
        )

        # Build action mask
        action_mask = self._build_action_mask(
            has_dice_on_table=False,
            can_roll=True,
            can_bank=(current_score > 0 and (total_score > 0 or current_score >= 1000)),
        )

        # Predict action
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        action = int(action)

        # Action 9 = roll, action 0 = bank
        return action == 9

    def actions(self, die_rolls: list[int]) -> list[str]:
        """Decide which dice to take after rolling.

        Args:
            die_rolls: List of dice values from the roll

        Returns:
            List of action strings for the Scorer (e.g., ["take_thousand_combo", "take_one"])
        """
        # Update internal dice state
        self.dice = die_rolls.copy()
        self.just_rolled = True

        # Check if we cached this exact roll
        dice_tuple = tuple(sorted(die_rolls))
        if self._cached_dice == dice_tuple and self._cached_action is not None:
            action = self._cached_action
        else:
            # Build observation with the rolled dice
            obs = self._build_observation_with_dice(die_rolls)

            # Build action mask based on available dice
            action_mask = self._build_action_mask_with_dice(die_rolls)

            # Predict action
            action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
            action = int(action)

            # Cache for potential reuse
            self._cached_action = action
            self._cached_dice = dice_tuple

        # Clear cache after use
        self._cached_action = None
        self._cached_dice = None

        # Translate model action to scorer method calls
        return self._action_to_scorer_methods(action, die_rolls)

    def _build_observation(
        self,
        dice_remaining: int,
        must_roll: bool,
        unbanked_score: int,
    ) -> NDArray[np.int64]:
        """Build observation vector (25 dims) for decision without dice on table."""
        obs = np.zeros(25, dtype=np.int64)

        # Positions 0-5: dice remaining (one-hot)
        if dice_remaining > 0 and dice_remaining <= 6:
            obs[dice_remaining - 1] = 1

        # Positions 6-13: available scoring actions (none if must roll)
        # Position 14: needs roll flag
        if must_roll or dice_remaining == 6:
            obs[14] = 1  # Must roll
        else:
            obs[14] = 0  # Can take actions
            # Mark that we can bank if we have unbanked score
            if unbanked_score > 0:
                obs[6] = 1  # Bank action available (position 6 = action 0 + 6)
            # Mark roll action
            obs[14] = 0  # Actually, let me check the encoding again

        # Positions 15-19: agent score bucket (one-hot)
        obs[15 + score_to_bucket(self.agent_score)] = 1

        # Positions 20-24: opponent score bucket (one-hot)
        obs[20 + score_to_bucket(self.opponent_score)] = 1

        return obs

    def _build_observation_with_dice(self, dice: list[int]) -> NDArray[np.int64]:
        """Build observation vector after rolling dice."""
        obs = np.zeros(25, dtype=np.int64)

        # Count remaining dice (non-zero)
        remaining = len([d for d in dice if d != 0])

        # Positions 0-5: dice remaining (one-hot)
        if remaining > 0:
            obs[remaining - 1] = 1

        # Positions 6-13: available scoring actions
        # Build legal actions from dice
        legal_actions = self._get_legal_actions(dice)

        if legal_actions == [9]:  # Must roll (blown)
            obs[14] = 1
        else:
            for action in legal_actions:
                if action <= 8:  # Actions 0-8 map to positions 6-14
                    obs[action + 6] = 1
            obs[14] = 0  # Not must-roll

        # Positions 15-19: agent score bucket (one-hot)
        obs[15 + score_to_bucket(self.agent_score)] = 1

        # Positions 20-24: opponent score bucket (one-hot)
        obs[20 + score_to_bucket(self.opponent_score)] = 1

        return obs

    def _build_action_mask(
        self,
        has_dice_on_table: bool,
        can_roll: bool,
        can_bank: bool,
    ) -> NDArray[np.bool_]:
        """Build action mask for decision without dice."""
        mask = np.zeros(10, dtype=bool)

        if can_bank:
            mask[0] = True  # Bank
        if can_roll:
            mask[9] = True  # Roll

        return mask

    def _build_action_mask_with_dice(self, dice: list[int]) -> NDArray[np.bool_]:
        """Build action mask based on available dice."""
        mask = np.zeros(10, dtype=bool)
        legal_actions = self._get_legal_actions(dice)

        for action in legal_actions:
            mask[action] = True

        return mask

    def _get_legal_actions(self, dice: list[int]) -> list[int]:
        """Get legal actions from current dice state."""
        scorer = Scorer(dice)

        if scorer.is_blown():
            return [9]  # Must roll

        actions: list[int] = []

        # Can bank if we have unbanked score and meet minimum requirement
        can_bank = self.agent_score > 0 or self.unbanked_score >= 1000
        if self.unbanked_score > 0 and can_bank:
            actions.append(0)

        # Check for triples
        if scorer.has_thousand_combo():
            actions.append(1)
        if scorer.has_two_hundred_combo():
            actions.append(2)
        if scorer.has_three_hundred_combo():
            actions.append(3)
        if scorer.has_four_hundred_combo():
            actions.append(4)
        if scorer.has_five_hundred_combo():
            actions.append(5)
        if scorer.has_six_hundred_combo():
            actions.append(6)

        # Check for singles
        if scorer.has_fifties():
            actions.append(7)
        if scorer.has_ones():
            actions.append(8)

        # Can always roll if we haven't just rolled
        if not self.just_rolled:
            actions.append(9)

        return actions

    def _action_to_scorer_methods(self, action: int, dice: list[int]) -> list[str]:
        """Translate model action to scorer method names.

        Args:
            action: Model action (0-9)
            dice: Current dice state

        Returns:
            List of scorer method names
        """
        # Action 0 (bank) and action 9 (roll) don't translate to scorer methods
        # They're handled by the game loop in should_roll

        if action == 1:
            return ["take_thousand_combo"]
        elif action == 2:
            return ["take_two_hundred_combo"]
        elif action == 3:
            return ["take_three_hundred_combo"]
        elif action == 4:
            return ["take_four_hundred_combo"]
        elif action == 5:
            return ["take_five_hundred_combo"]
        elif action == 6:
            return ["take_six_hundred_combo"]
        elif action == 7:
            # Take one 5 (fifty)
            return ["take_fifty"]
        elif action == 8:
            # Take one 1
            return ["take_one"]
        else:
            # This shouldn't happen - actions 0 and 9 shouldn't reach here
            # But if they do, return empty (no dice to take)
            return []
