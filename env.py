import cv2
import pyautogui
import numpy as np
import os
import torch
import torchvision.transforms as T

from window import WindowInterface

class DemineurInterface():
    
    def __init__(self, rows = 16, cols = 30):
        
        self.windowInterface = WindowInterface("Démineur")
        self.windowInterface.move_and_resize(w=500,h=320)
        self.grid = Grid(rows, cols)
        self.action_space_nb = self.grid.action_space_nb
        self.victories = 0

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(7, 7)
        ])
    
    def grab_image(self):
        
        img = self.windowInterface.screenshot()[:,:,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def reset(self):
        self.grid.reset_grid()
        #print("reset")
        
        
    
    def get_state(self):
        
        img = self.grab_image()
        grid = self.grid.extract_from_image(img)

        return torch.unsqueeze(self.transform(grid), 0).to(torch.float32)

        return torch.unsqueeze(torch.unsqueeze(torch.Tensor(grid), 0), 0)        
    
    def send_control(self, x, y):
        pyautogui.click(x, y)
        
    def step(self, index, cpt = 1):
               
        i = index // self.grid.cols
        j = index % self.grid.cols

        x, y = self.grid.click_pos[i,j]
        
        if not self.grid.is_free(i,j):
            #print(f"Case {i},{j} is used")
            pyautogui.moveTo(x, y)
            return 0, False # Reward, Done
        
        isolated = self.grid.is_isolated(i,j)
        
        
        self.send_control(x, y)
        
        img = self.grab_image()

        # for debugging
        #cv2.imwrite("output/{}.png".format(cpt), img)

        done = self.grid.is_done(img)
        
        #print(f"Click Case {i},{j}. done:", done)
        
        if done == 0:
            # If not over, since the action was clicking on an empty case, it's not over

            if isolated:
                # Try to influence the model to click near already known cases
                return .7, False

            return 1, False  
        elif done == 1:
            # Victory
            self.victories += 1
            return 1, True
        else:
            # Click on a bomb
            return 0, True
        

class Grid():
    
    def __init__(self, rows=16, cols=30):

        self.rows = rows
        self.cols = cols
        self.case_width = 16
        self.case_height = 16
        self.first_case_x = 55
        self.first_case_y = 12
        
        self.status_width = 26
        self.status_height = 26
        self.status_i = 15

        assert self.cols in [9, 16, 30], "Unknown Pattern"
        if self.cols == 9:
            self.status_j = 71
        elif self.cols == 16:
            self.status_j = 127
        else:
            self.status_j = 239
        
        self.window_offset_i = 46
        self.window_offset_j = 1
        
        # Define click position for each case on the grid
        X = np.floor(np.linspace(self.window_offset_i + self.first_case_x + self.case_height // 2, self.window_offset_i + self.first_case_x + self.rows * self.case_height - self.case_height // 2, self.rows))
        Y = np.floor(np.linspace(self.window_offset_j + self.first_case_y + self.case_width // 2, self.window_offset_j + self.first_case_y + self.cols * self.case_width - self.case_width // 2, self.cols))

        self.click_pos = np.stack(np.meshgrid(Y,X), axis=-1)
        
        # Load templates
        self.load_cases()
        self.load_status()
        
        self.action_space_nb = self.rows * self.cols
        
        self.reset_grid()
        
    def reset_grid(self):
        # Twice to get the focus
        pyautogui.press("enter")
        pyautogui.press("enter")
        pyautogui.click(self.window_offset_j + self.status_j + self.status_height // 2, self.window_offset_i + self.status_i + self.status_width // 2)
        #pyautogui.click(self.window_offset_j + self.status_j + self.status_height // 2, self.window_offset_i + self.status_i + self.status_width // 2)
        #pyautogui.press('F2')
        
        self.grid = np.zeros((self.rows, self.cols)) + self.cases_labels.index("raw")
        self.done = 0

    def load_cases(self):
        
        casesNames = os.listdir("images/cases")
        
        self.cases_templates = np.zeros((len(casesNames), self.case_width, self.case_height, 3))
        self.cases_labels = []
        
        for i in range(len(casesNames)):
            
            case = cv2.imread("images/cases/" + casesNames[i])[:,:,:3]
            
            self.cases_templates[i,...] = case[:,:,::-1]
            self.cases_labels.append(casesNames[i].rsplit('_', 1)[0])
    
    def load_status(self):
        
        statusNames = os.listdir("images/status")
        
        self.status_templates = np.zeros((len(statusNames), self.status_width, self.status_height, 3))
        self.status_labels = []
        
        for i in range(len(statusNames)):
            
            status = cv2.imread("images/status/" + statusNames[i])[:,:,:3]
            
            self.status_templates[i,...] = status[:,:,::-1]
            self.status_labels.append(statusNames[i].split('.')[0])
        
        
    def extract_from_image(self, img):
        """ Convert the grid on the image into a python object """
        
        for i in range(self.rows):
            for j in range(self.cols):
                
                x = self.first_case_x + i * self.case_height
                y = self.first_case_y + j * self.case_width
                
                case_extract = img[x:x + self.case_height, y:y + self.case_width]

                self.grid[i,j] = self.get_matching_template(case_extract, self.cases_templates)

        self.done = self.is_done(img)
        
        return self.grid

    def get_matching_template(self, input, templates):
        
        input = np.tile(input[None,...], [len(templates), 1, 1, 1])
        diff = np.sum((templates - input)**2, axis=(1,2,3))
        
        return np.argmin(diff)

    def is_done(self, img):
                
        img = img.astype(np.float32)

        maxi = 0
        iRef = 0

        for i in range(len(self.status_templates)):

            template = self.status_templates[i].astype(np.float32)

            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)

            if res.max() > maxi:
                maxi = res.max()
                iRef = i

        if iRef == self.status_labels.index("continue"):
            return 0
        elif iRef == self.status_labels.index("victory"):
            return 1
        else:
            return 2
            
        
    def is_free(self, i, j):
        
        return self.grid[i,j] == self.cases_labels.index("raw")

    def is_isolated(self, i, j):

        mask = np.zeros((3,3)) + self.cases_labels.index("raw")

        grid_pad = np.copy(self.grid)
        grid_pad = np.pad(grid_pad, 1, constant_values=self.cases_labels.index("raw"))
        return (grid_pad[i:i+3, j:j+3] == mask).all()
    
if __name__ == "__main__":

    env = DemineurInterface(9, 9)

    img = env.grab_image()
    print(env.grid.is_done(img))