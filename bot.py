import cv2 as cv
import pyautogui
from time import sleep, time
from threading import Thread, Lock
from math import sqrt


class BotState:
    INITIALIZING = 0
    SEARCHING = 1
    MOVING = 2
    MINING = 3
    BANKING = 4
    BACKTRACKING = 5


class RSBot:
    
    # constants
    INITIALIZING_SECONDS = 6
    MINING_SECONDS = 20
    MOVEMENT_STOPPED_THRESHOLD = 0.975
    IGNORE_RADIUS = 130
    TOOLTIP_MATCH_THRESHOLD = 0.55
    BANKINGPERIOD = 40

    # threading variables
    stopped = True
    lock = None

    # variables
    state = None
    targets = []
    screenshot = None
    timestamp = None
    bankingTimestamp = None
    movement_screenshot = None
    window_offset = (0,0)
    window_w = 0
    window_h = 0
    tree_tooltip = None
    bank_tooltip = None
    click_history = []

    def __init__(self, window_offset, window_size):
        # create a thread lock object
        self.lock = Lock()

        # used to determine where to move mouse
        self.window_offset = window_offset
        self.window_w = window_size[0]
        self.window_h = window_size[1]

        # load the tooltip images used to confirm our object detection
        self.tree_tooltip = cv.imread('treeTooltip.jpg', 0)
        self.bank_tooltip = cv.imread('bankTooltip.jpg', 0)

        # start bot in the initializing state to allow time to setup. Begin timer and bank timer
        self.state = BotState.INITIALIZING
        self.timestamp = time()
        self.bankingTimestamp = time()

    def click_next_target(self):
        targets = self.targets_ordered_by_distance(self.targets)

        target_i = 0
        found_resource = False
        while not found_resource and target_i < len(targets):
            # if we stopped our script, exit this loop
            if self.stopped:
                break

            # look at next target in list
            target_pos = targets[target_i]
            screen_x, screen_y = self.get_screen_position(target_pos)
            print('Moving mouse to x:{} y:{}'.format(screen_x, screen_y))

            # move the mouse
            pyautogui.moveTo(x=screen_x, y=screen_y)
            # short pause to let the mouse movement complete and allow for tooltip to appear
            sleep(1.450)
            # confirm resource tooltip
            if self.confirm_tooltip(target_pos):
                print('Click on confirmed target at x:{} y:{}'.format(screen_x, screen_y))
                found_resource = True
                pyautogui.click()
                # save this position to the click history
                self.click_history.append(target_pos)
            target_i += 1

        return found_resource

    def click_next_bank(self):
        targets = self.targets_ordered_by_distance(self.targets)

        target_i = 0
        found_bank = False
        while not found_bank and target_i < len(targets):
            # if we stopped our script, exit this loop
            if self.stopped:
                break

            target_pos = targets[target_i]
            screen_x, screen_y = self.get_screen_position(target_pos)
            print('Moving mouse to x:{} y:{}'.format(screen_x, screen_y))

            # move the mouse
            pyautogui.moveTo(x=screen_x+280, y=screen_y+400)
            # short pause to let the mouse movement complete and tooltip to appear
            sleep(1.450)
            # confirm bank tooltip
            if self.confirm_bank_tooltip(target_pos):
                print('Click on confirmed bank at x:{} y:{}'.format(screen_x, screen_y))
                found_bank = True
                pyautogui.click()
                # save this position to the click history
                self.click_history.append(target_pos)
            target_i += 1

        return found_bank

    def have_stopped_moving(self):
        # if we haven't stored a screenshot to compare to, do that first
        if self.movement_screenshot is None:
            self.movement_screenshot = self.screenshot.copy()
            return False

        # compare the old screenshot to the new screenshot
        result = cv.matchTemplate(self.screenshot, self.movement_screenshot, cv.TM_CCOEFF_NORMED)
        # images are same size
        similarity = result[0][0]
        print('Movement detection similarity: {}'.format(similarity))

        if similarity >= self.MOVEMENT_STOPPED_THRESHOLD:
            # pictures look similar, so we've probably stopped moving
            print('Movement detected stop')
            return True

        # still moving update old screenshot
        self.movement_screenshot = self.screenshot.copy()
        return False

    def targets_ordered_by_distance(self, targets):
        # our character is always in the center of the screen
        my_pos = (self.window_w / 2, self.window_h / 2)
        def pythagorean_distance(pos):
            return sqrt((pos[0] - my_pos[0])**2 + (pos[1] - my_pos[1])**2)
        targets.sort(key=pythagorean_distance)

        # ignore targets at are too close to our character (within 130 pixels) to avoid 
        # re-clicking a deposit we just mined
        targets = [t for t in targets if pythagorean_distance(t) > self.IGNORE_RADIUS]

        return targets

    def confirm_tooltip(self, target_position):
        # check the current screenshot for the resource tooltip using match template
        grayscale = cv.cvtColor(self.screenshot, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grayscale, self.tree_tooltip, cv.TM_CCOEFF_NORMED)
        # get the best match postition
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # if matching tooltip threshold object found
        if max_val >= self.TOOLTIP_MATCH_THRESHOLD:
            print('Tooltip found in image at {}'.format(max_loc))
            return True
        print('Tooltip not found.')
        return False

    def confirm_bank_tooltip(self, target_position):
        # check the current screenshot for the bank tooltip using match template
        grayscale = cv.cvtColor(self.screenshot, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grayscale, self.bank_tooltip, cv.TM_CCOEFF_NORMED)
        # get the best match postition
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # if matching tooltip threshold object found
        if max_val >= self.TOOLTIP_MATCH_THRESHOLD:
            print('Bank Tooltip found in image at {}'.format(max_loc))
            return True
        print('Bank Tooltip not found.')
        return False

    def click_backtrack(self):
        # geting most recent movement from stack
        last_click = self.click_history.pop()
        # have to mirror last click to reverse movement
        my_pos = (self.window_w / 2, self.window_h / 2)
        mirrored_click_x = my_pos[0] - (last_click[0] - my_pos[0])
        mirrored_click_y = my_pos[1] - (last_click[1] - my_pos[1])
        # convert this screenshot position to a screen position
        screen_x, screen_y = self.get_screen_position((mirrored_click_x, mirrored_click_y))
        print('Backtracking to x:{} y:{}'.format(screen_x, screen_y))
        pyautogui.moveTo(x=screen_x, y=screen_y)
        # short pause to let the mouse movement complete
        sleep(0.500)
        pyautogui.click()

    # determine screen position of objects
    def get_screen_position(self, pos):
        return (pos[0] + self.window_offset[0], pos[1] + self.window_offset[1])

    # threading methods

    def update_targets(self, targets):
        self.lock.acquire()
        self.targets = targets
        self.lock.release()

    def update_screenshot(self, screenshot):
        self.lock.acquire()
        self.screenshot = screenshot
        self.lock.release()

    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.stopped = True

    # main bot logic
    def run(self):
        while not self.stopped:
            if self.state == BotState.INITIALIZING:
                # do no bot actions until the startup waiting period is complete
                if time() > self.timestamp + self.INITIALIZING_SECONDS:
                    # start searching when the waiting period is over
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()

            elif self.state == BotState.BANKING:
                               
                sleep(.6)
                # continue searching for bank until found
                success = False
                while not success:
                    success = self.click_next_bank()

                # wait for move and bank open
                sleep(9)

                # deposit inventory and close bank
                pyautogui.press('3')
                sleep(0.8)
                pyautogui.press('esc')

                # ready to gather more
                self.bankingTimestamp = time()
                self.lock.acquire()
                self.state = BotState.SEARCHING
                self.lock.release()

            elif self.state == BotState.SEARCHING:
                # check the given click point targets, confirm a resource then use it.
                success = self.click_next_target()
                # if not successful, try one more time
                if not success:
                    success = self.click_next_target()

                # if successful, switch state to moving
                if success:
                    self.lock.acquire()
                    self.state = BotState.MOVING
                    self.lock.release()
                #unsuccessful so try to backtrack
                elif len(self.click_history) > 0:
                    self.click_backtrack()
                    self.lock.acquire()
                    self.state = BotState.BACKTRACKING
                    self.lock.release()
                else:
                    # stay in place and keep searching
                    pass

            elif self.state == BotState.MOVING or self.state == BotState.BACKTRACKING:
                # determine if we have stopped moving
                if not self.have_stopped_moving():
                    # wait a short time to allow for the character position to change
                    sleep(0.500)
                else:
                    # get new time if beginning mining so we know when we started mining
                    self.lock.acquire()
                    if self.state == BotState.MOVING:
                        self.timestamp = time()
                        self.state = BotState.MINING
                    # backtracking
                    elif self.state == BotState.BACKTRACKING:
                        self.state = BotState.SEARCHING
                    self.lock.release()
                
            elif self.state == BotState.MINING:
                # waiting for mining to be done and checking if a banking period has elapsed
                if time() > self.timestamp + self.MINING_SECONDS:
                    # bank if necessary otherwise continue searching
                    if time() >= self.bankingTimestamp + self.BANKINGPERIOD:
                        self.lock.acquire()
                        self.state = BotState.BANKING
                        self.lock.release()
                    else:
                        self.lock.acquire()
                        self.state = BotState.SEARCHING
                        self.lock.release()
