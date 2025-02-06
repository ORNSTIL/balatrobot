#!/usr/bin/python3

import os
import sys
import json
import socket
import time
from enum import Enum
from gamestates import cache_state
import subprocess
import random
import platform
from pathlib import Path
import abc


class State(Enum):
    SELECTING_HAND = 1
    HAND_PLAYED = 2
    DRAW_TO_HAND = 3
    GAME_OVER = 4
    SHOP = 5
    PLAY_TAROT = 6
    BLIND_SELECT = 7
    ROUND_EVAL = 8
    TAROT_PACK = 9
    PLANET_PACK = 10
    MENU = 11
    TUTORIAL = 12
    SPLASH = 13
    SANDBOX = 14
    SPECTRAL_PACK = 15
    DEMO_CTA = 16
    STANDARD_PACK = 17
    BUFFOON_PACK = 18
    NEW_ROUND = 19


class Actions(Enum):
    SELECT_BLIND = 1
    SKIP_BLIND = 2
    PLAY_HAND = 3
    DISCARD_HAND = 4
    END_SHOP = 5
    REROLL_SHOP = 6
    BUY_CARD = 7
    BUY_VOUCHER = 8
    BUY_BOOSTER = 9
    SELECT_BOOSTER_CARD = 10
    SKIP_BOOSTER_PACK = 11
    SELL_JOKER = 12
    USE_CONSUMABLE = 13
    SELL_CONSUMABLE = 14
    REARRANGE_JOKERS = 15
    REARRANGE_CONSUMABLES = 16
    REARRANGE_HAND = 17
    PASS = 18
    START_RUN = 19
    SEND_GAMESTATE = 20


class Bot:
    def __init__(
        self,
        deck: str,
        stake: int = 1,
        seed: str | None = None,
        challenge: str | None = None,
        bot_port: int = 12346,
    ):
        self.G = None
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.challenge = challenge

        self.bot_port = bot_port

        self.addr = ("localhost", self.bot_port)
        self.running = False
        self.balatro_instance = None

        self.sock = None

        self.state = {}

    @abc.abstractmethod
    def skip_or_select_blind(self, G):
        raise NotImplementedError(
            "Error: Bot.skip_or_select_blind must be implemented."
        )

    @abc.abstractmethod
    def select_cards_from_hand(self, G):
        raise NotImplementedError(
            "Error: Bot.select_cards_from_hand must be implemented."
        )

    @abc.abstractmethod
    def select_shop_action(self, G):
        raise NotImplementedError("Error: Bot.select_shop_action must be implemented.")

    @abc.abstractmethod
    def select_booster_action(self, G):
        raise NotImplementedError(
            "Error: Bot.select_booster_action must be implemented."
        )

    @abc.abstractmethod
    def sell_jokers(self, G):
        raise NotImplementedError("Error: Bot.sell_jokers must be implemented.")

    @abc.abstractmethod
    def rearrange_jokers(self, G):
        raise NotImplementedError("Error: Bot.rearrange_jokers must be implemented.")

    @abc.abstractmethod
    def use_or_sell_consumables(self, G):
        raise NotImplementedError(
            "Error: Bot.use_or_sell_consumables must be implemented."
        )

    @abc.abstractmethod
    def rearrange_consumables(self, G):
        raise NotImplementedError(
            "Error: Bot.rearrange_consumables must be implemented."
        )

    @abc.abstractmethod
    def rearrange_hand(self, G):
        raise NotImplementedError("Error: Bot.rearrange_hand must be implemented.")

    # Attempts to start balatro with resonable defaults for environment
    def start_balatro_instance(self):
        if platform.system() == "Linux":
            balatro_exec_path = os.path.expandvars(
                "$HOME/.local/share/Steam/steamapps/common/Balatro/Balatro.exe"
            )
            client_install_path = os.path.expandvars("$HOME/.local/share/Steam")
            data_path = os.path.expandvars(
                "$HOME/.local/share/Steam/steamapps/compatdata/2379780"
            )
            PROTON_VERSION = "GE-Proton9-4"
            proton_path = os.path.expandvars(
                f"$HOME/.local/share/Steam/compatibilitytools.d/{PROTON_VERSION}/proton"
            )
            self.balatro_instance = subprocess.Popen(
                [
                    "/usr/bin/env",
                    "WINEDLLOVERRIDES=version=n,b",
                    f"STEAM_COMPAT_CLIENT_INSTALL_PATH={client_install_path}",
                    f"STEAM_COMPAT_DATA_PATH={data_path}",
                    proton_path,
                    "waitforexitandrun",
                    balatro_exec_path,
                    str(self.bot_port),
                ]
            )
        elif platform.system() == "Windows":
            balatro_exec_path = (
                r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
            )
            self.balatro_instance = subprocess.Popen(
                [balatro_exec_path, str(self.bot_port)]
            )
        else:
            raise RuntimeError("Unknown platform for staring balatro.")

    def stop_balatro_instance(self):
        if self.balatro_instance:
            self.balatro_instance.kill()

    def sendcmd(self, cmd, **kwargs):
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def actionToCmd(self, action):
        result = []

        for x in action:
            if isinstance(x, Actions):
                result.append(x.name)
            elif type(x) is list:
                result.append(",".join([str(y) for y in x]))
            else:
                result.append(str(x))

        return "|".join(result)

    def verifyimplemented(self):
        try:
            self.skip_or_select_blind({})
            self.select_cards_from_hand({})
            self.select_shop_action({})
            self.select_booster_action({})
            self.sell_jokers({})
            self.rearrange_jokers({})
            self.use_or_sell_consumables({})
            self.rearrange_consumables({})
            self.rearrange_hand({})
        except NotImplementedError as e:
            print(e)
            sys.exit(0)
        except:
            pass

    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))

    def chooseaction(self):
        if self.G["state"] == State.GAME_OVER:
            print("ending game")
            self.running = False

        match self.G["waitingFor"]:
            case "start_run":
                seed = self.seed
                if seed is None:
                    seed = self.random_seed()
                return [
                    Actions.START_RUN,
                    self.stake,
                    self.deck,
                    seed,
                    self.challenge,
                ]
            case "skip_or_select_blind":
                return self.skip_or_select_blind(self.G)
            case "select_cards_from_hand":
                return self.select_cards_from_hand(self.G)
            case "select_shop_action":
                return self.select_shop_action(self.G)
            case "select_booster_action":
                return self.select_booster_action(self.G)
            case "sell_jokers":
                return self.sell_jokers(self.G)
            case "rearrange_jokers":
                return self.rearrange_jokers(self.G)
            case "use_or_sell_consumables":
                return self.use_or_sell_consumables(self.G)
            case "rearrange_consumables":
                return self.rearrange_consumables(self.G)
            case "rearrange_hand":
                return self.rearrange_hand(self.G)

    def run_step(self):
        if self.sock is None:
            self.verifyimplemented()
            self.state = {}
            self.G = None

            self.running = True
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1)
            self.sock.connect(self.addr)

        if self.running:
            self.sendcmd("HELLO")

            jsondata = {}
            try:
                data = self.sock.recv(65536)
                jsondata = json.loads(data)

                if "response" in jsondata:
                    print(jsondata["response"])
                else:
                    self.G = jsondata
                    if self.G["waitingForAction"]:
                        cache_state(self.G["waitingFor"], self.G)
                        action = self.chooseaction()
                        if action == None:
                            raise ValueError("All actions must return a value!")

                        cmdstr = self.actionToCmd(action)
                        self.sendcmd(cmdstr)
            except socket.error as e:
                print(e)
                print("Socket error, reconnecting...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(1)
                self.sock.connect(self.addr)

    def run(self):
        self.run_step()
        while self.running:
            self.run_step()
