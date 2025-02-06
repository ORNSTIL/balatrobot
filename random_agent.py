import random
from bot import Bot, Actions
import time


class RandomBot(Bot):

    def skip_or_select_blind(self, G):
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, G):
        hand = self.G["hand"]
        num_cards = min(random.randrange(1, 5), len(hand))
        cards = random.sample(range(len(hand)), num_cards)
        print(f"selecting {cards}")
        return [Actions.PLAY_HAND, cards]

    def select_shop_action(self, G):
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, []]


if __name__ == "__main__":
    bot = RandomBot(deck="Blue Deck", stake=1, seed=None, challenge=None, bot_port=12348)
    bot.start_balatro_instance()
    time.sleep(20)
    bot.run()
