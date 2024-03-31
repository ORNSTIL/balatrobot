
--local Hook = require "hook"
--local Bot = require "bot"

Middleware = { }

function random_key(tb)
    local keys = {}
    for k in pairs(tb) do table.insert(keys, k) end
    return keys[math.random(#keys)]
end

function random_element(tb)
    local keys = {}
    for k in pairs(tb) do table.insert(keys, k) end
    return tb[keys[math.random(#keys)]]
end

function Middleware.add_event_sequence_recursive(events)
    if events == nil or #events <= 0 then
        return true
    end
    
    local head = table.remove(events, 1)

    local _event = Event({
        trigger = 'after',
        delay = head.delay,
        blocking = false,
        func = function()
            head.func(head.args)
            Middleware.add_event_sequence_recursive(events)
            return true
        end
    })
    G.E_MANAGER:add_event(_event)
end

function Middleware.add_event_sequence(events)

    local _lastevent = nil
    local _totaldelay = 0.0

    for k, event in pairs(events) do
        _totaldelay = _totaldelay + event.delay

        local _event = Event({
            trigger = 'after',
            delay = _totaldelay,
            blocking = false,
            func = function()
                event.func(event.args)
                return true
            end
        })
        G.E_MANAGER:add_event(_event)
        _lastevent = _event
    end

    return _lastevent
end

Middleware.queuedactions = List.new()
Middleware.currentaction = nil

Middleware.firewhenready = { }

local function firewhenready(condition, func)
    for i = 1, #Middleware.firewhenready, 1 do
        if Middleware.firewhenready[i] == nil then
            Middleware.firewhenready[i] = {
                ready = condition,
                fire = func
            }
            return nil
        end
    end

    Middleware.firewhenready[#Middleware.firewhenready + 1] = {
        ready = condition,
        fire = func
    }
end

local function queueaction(func, delay)

    if not delay then
        delay = Bot.SETTINGS.action_delay
    end

    List.pushleft(Middleware.queuedactions, { func = func, delay = delay })
end

local function pushbutton(button, delay)
    queueaction(function()
        if button and button.config and button.config.button then
            G.FUNCS[button.config.button](button)
        end
    end, delay)
end

local function clickcard(card, delay)
    queueaction(function()
        if card and card.click then
            card:click()
        end
    end, delay)
end

local function usecard(card, delay)

    queueaction(function()
        firewhenready(function()
            if not card then return true end
            return card.children.use_button or card.children.buy_and_use_button or card.children.buy_button or card.children.use_and_sell_button
        end, function()
            local _use_button = nil
            local _use_button = card.children.use_button and card.children.use_button.definition
            if _use_button and _use_button.config.button == nil then
                local _node_index = card.ability.consumeable and 2 or 1
                _use_button = _use_button.nodes[_node_index]
            end
            local _buy_and_use_button = card.children.buy_and_use_button and card.children.buy_and_use_button.definition
            local _buy_button = card.children.buy_button and card.children.buy_button.definition
            local _sell_button = card.children.use_and_sell_button and card.children.use_and_sell_button.definition
    
            if _use_button then
                pushbutton(_use_button, delay)
            elseif _buy_and_use_button then
                pushbutton(_buy_and_use_button, delay)
            elseif _buy_button then
                pushbutton(_buy_button, delay)
            elseif _sell_button then
                pushbutton(_sell_button, delay)
            end
        end)
    end, 0.0)
end

Middleware.prev_gamestate = -1

Middleware.BUTTONS = {

    -- Shop Phase Buttons
    NEXT_ROUND = nil,
    REROLL = nil,

    -- Pack Phase Buttons
    SKIP_PACK = nil,

}

local function c_update()

    -- Process the queue of Bot events
    if not List.isempty(Middleware.queuedactions) and
        (not Middleware.currentaction or 
            (Middleware.currentaction and Middleware.currentaction.complete)) then

        local _func_and_delay = List.popright(Middleware.queuedactions)

        local _event = Middleware.add_event_sequence({
            { func = _func_and_delay.func, delay = _func_and_delay.delay }
        })
        Middleware.currentaction = _event
    end

    -- Run functions that have been waiting for a condition to be met
    for i = 1, #Middleware.firewhenready, 1 do
        if Middleware.firewhenready[i] and Middleware.firewhenready[i].ready() then
            Middleware.firewhenready[i].fire()
            Middleware.firewhenready[i] = nil
        end
    end
end

local function c_onmainmenu()

    local function click_run_play_button()
        G.FUNCS.start_run(nil, {stake = Bot.SETTINGS.stake, seed = Bot.SETTINGS.seed, challenge = Bot.SETTINGS.challenge})
        return true
    end

    local function click_main_play_button()
        local _play_button = G.MAIN_MENU_UI:get_UIE_by_ID('main_menu_play')

        G.FUNCS[_play_button.config.button]({
            config = { }
        })
        G.FUNCS.exit_overlay_menu()

        return true  

    end

    Middleware.add_event_sequence_recursive({
        { func = click_main_play_button, delay = 3.0 },
        { func = click_run_play_button, delay = 1.0 }
    })
end

local function c_can_play_hand()

    local function click_play_hand()
        if G.buttons == nil then return true end

        local _play_button = UIBox:get_UIE_by_ID('play_button', G.buttons.UIRoot)
        if _play_button ~= nil and _play_button.config.button ~= nil then
           G.FUNCS[_play_button.config.button](_play_button)
        end
    end

    local function click_discard_hand()
        if G.buttons == nil then return true end

        local _discard_button = UIBox:get_UIE_by_ID('discard_button', G.buttons.UIRoot)
        if _discard_button ~= nil and _discard_button.config.button ~= nil then
            G.FUNCS[_discard_button.config.button](_discard_button)
        end
    end

    local function decide()

        local _action, _cards_to_play = Bot.select_cards_from_hand()

        local _events = { }
        for i = 1, #_cards_to_play, 1 do
            _events[i] = {
                func = function()
                    G.hand.cards[i]:click()
                end,
                delay = 0.5
            }
        end

        -- Option 1: Play Hand
        if _action == Bot.CHOICES.PLAY_HAND then
            _events[#_events+1] = { func = click_play_hand, delay = 2.0 }
        end

        -- Option 2: Discard Hand
        if _action == Bot.CHOICES.DISCARD_HAND then
            _events[#_events+1] = { func = click_discard_hand, delay = 2.0 } 
        end

        Middleware.add_event_sequence_recursive(_events)

    end

    decide()

end

local function c_select_blind()

    local _blind_on_deck = G.GAME.blind_on_deck

    local _blind_obj = G.blind_select_opts[string.lower(_blind_on_deck)]
    local _select_button = _blind_obj:get_UIE_by_ID('select_blind_button')


    if _blind_on_deck == 'Boss' then
        pushbutton(_select_button)
        return
    end

    local _skip_button = _blind_obj:get_UIE_by_ID('tag_'.._blind_on_deck).children[2]

    local _choice = Bot.skip_or_select_blind(_blind_on_deck)

    local _button = nil
    if _choice == Bot.CHOICES.SELECT_BLIND then
        _button = _select_button
    elseif _choice == Bot.CHOICES.SKIP_BLIND_SELECT_VOUCHER then
        _button = _skip_button
    end

    pushbutton(_button)
end

local function c_can_choose_booster_cards()

    if not G.pack_cards.cards then return end

    local _action, _card, _hand_cards = Bot.select_booster_action(G.pack_cards.cards, G.hand.cards)

    if _action == Bot.CHOICES.SKIP_BOOSTER_PACK then
        pushbutton(Middleware.BUTTONS.SKIP_PACK)
    elseif _action == Bot.CHOICES.SELECT_BOOSTER_CARD then

        -- Click each card from your deck first (only occurs if _pack_card is consumeable)
        for i = 1, #_hand_cards, 1 do
            clickcard(_hand_cards[i])
        end

        -- Then select the booster card to activate
        clickcard(_card)
        usecard(_card)
    end

    if G.GAME.pack_choices - 1 > 0 then
        queueaction(function()
            firewhenready(function()
                return G.STATE_COMPLETE
            end, c_can_choose_booster_cards)
        end, 0.0)
    else
        if G.GAME.PACK_INTERRUPT == G.STATES.BLIND_SELECT then
            queueaction(function()
                firewhenready(function()
                    return G.STATE_COMPLETE and G.STATE == G.STATES.BLIND_SELECT
                end, c_select_blind)
            end, 0.0)
        end
    end
    
end


local function c_can_shop()

    local _done_shopping = false

    local _b_can_round_end_shop = true
    local _b_can_reroll_shop = Middleware.BUTTONS.REROLL and Middleware.BUTTONS.REROLL.config and Middleware.BUTTONS.REROLL.config.button

    local _cards_to_buy = { }
    for i = 1, #G.shop_jokers.cards, 1 do
        _cards_to_buy[i] = G.shop_jokers.cards[i].cost <= G.GAME.dollars and G.shop_jokers.cards[i] or nil
    end

    local _vouchers_to_buy = { }
    for i = 1, #G.shop_vouchers.cards, 1 do
        _vouchers_to_buy[i] = G.shop_vouchers.cards[i].cost <= G.GAME.dollars and G.shop_vouchers.cards[i] or nil
    end

    local _boosters_to_buy = { }
    for i = 1, #G.shop_booster.cards, 1 do
        _boosters_to_buy[i] = G.shop_booster.cards[i].cost <= G.GAME.dollars and G.shop_booster.cards[i] or nil
    end

    local _choices = { }
    _choices[Bot.CHOICES.NEXT_ROUND_END_SHOP] = _b_can_round_end_shop
    _choices[Bot.CHOICES.REROLL_SHOP] = _b_can_reroll_shop
    _choices[Bot.CHOICES.BUY_CARD] = #_cards_to_buy > 0 and _cards_to_buy or nil
    _choices[Bot.CHOICES.BUY_VOUCHER] = #_vouchers_to_buy > 0 and _vouchers_to_buy or nil
    _choices[Bot.CHOICES.BUY_BOOSTER] = #_boosters_to_buy > 0 and _boosters_to_buy or nil
    
    local _action, _card = Bot.select_shop_action(_choices)

    if _action == Bot.CHOICES.NEXT_ROUND_END_SHOP then
        pushbutton(Middleware.BUTTONS.NEXT_ROUND)
        _done_shopping = true
    elseif _action == Bot.CHOICES.REROLL_SHOP then
        pushbutton(Middleware.BUTTONS.REROLL)
    elseif _action == Bot.CHOICES.BUY_CARD or _action == Bot.CHOICES.BUY_VOUCHER or _action == Bot.CHOICES.BUY_BOOSTER then
        _done_shopping = _action == Bot.CHOICES.BUY_BOOSTER

        clickcard(_card)
        usecard(_card)
    end

    if not _done_shopping then
        queueaction(function()
            firewhenready(function()
                return G.shop ~= nil and G.STATE_COMPLETE and G.STATE == G.STATES.SHOP
            end, c_can_shop)
        end)
    end
end


local function c_can_rearrange_jokers()
    Bot.rearrange_jokers()
end

local function c_can_rearrange_hand()
    Bot.rearrange_hand()
end

local function c_start_play_hand()
    Middleware.add_event_sequence_recursive({
        { func = c_can_rearrange_jokers, delay = 2.0 },
        { func = c_can_rearrange_hand, delay = 2.0 },
        { func = c_can_play_hand, delay = 2.0 }
    })
end

local function w_gamestate(...)
    local _t, _k, _v = ...

    if _k == 'STATE' and _v == G.STATES.MENU then
        c_onmainmenu()
    end
end

local function c_initgamehooks()

    -- Hooks break SAVE_MANAGER.channel:push so disable saving. Who needs it when you are botting anyway...
    G.SAVE_MANAGER = {
        channel = {
            push = function() end
        }
    }

    -- Detect when hand has been drawn
    G.GAME.blind.drawn_to_hand = Hook.addcallback(G.GAME.blind.drawn_to_hand, function(...)
        --Middleware.add_event_sequence_recursive({
         --   { func = c_can_rearrange_jokers, delay = 2.0 },
         --   { func = c_can_rearrange_hand, delay = 2.0 },
         --   { func = c_can_play_hand, delay = 2.0 }
       -- })
    end)

    -- Hook button snaps
    G.CONTROLLER.snap_to = Hook.addcallback(G.CONTROLLER.snap_to, function(...)
        local _self = ...

        if _self and _self.snap_cursor_to.node and _self.snap_cursor_to.node.config and _self.snap_cursor_to.node.config.button then
            sendDebugMessage("SNAPTO BUTTON: ".._self.snap_cursor_to.node.config.button)

            local _button = _self.snap_cursor_to.node
            local _buttonfunc = _self.snap_cursor_to.node.config.button

            if _buttonfunc == 'select_blind' and G.STATE == G.STATES.BLIND_SELECT then
                c_select_blind()
            elseif _buttonfunc == 'cash_out' then
                pushbutton(_button)
            elseif _buttonfunc == 'toggle_shop' and G.shop ~= nil then -- 'next_round_button'
                Middleware.BUTTONS.NEXT_ROUND = _button

                firewhenready(function()
                    return G.shop ~= nil and G.STATE_COMPLETE and G.STATE == G.STATES.SHOP
                end, c_can_shop)
            end
        end
    end)

    -- Hook Pack focus
    G.CONTROLLER.recall_cardarea_focus = Hook.addcallback(G.CONTROLLER.recall_cardarea_focus, function(...)
        local _self, _arg = ...
        if _arg == 'pack_cards' then
            firewhenready(function()
                return G.STATE_COMPLETE
            end, c_can_choose_booster_cards)
        end
    end)

    -- Set reroll availability
    G.FUNCS.can_reroll = Hook.addcallback(G.FUNCS.can_reroll, function(...)
        local _e = ...
        Middleware.BUTTONS.REROLL = _e
    end)

    -- Booster pack skip availability
    G.FUNCS.can_skip_booster = Hook.addcallback(G.FUNCS.can_skip_booster, function(...)
        local _e = ...
        Middleware.BUTTONS.SKIP_PACK = _e
    end)
end

function Middleware.hookbalatro()
    -- Start game from main menu
    G.start_run = Hook.addcallback(G.start_run, c_initgamehooks)
    G = Hook.addonwrite(G, w_gamestate)
    G.update = Hook.addcallback(G.update, c_update)
end

return Middleware