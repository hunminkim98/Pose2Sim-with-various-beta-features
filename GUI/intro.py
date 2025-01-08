import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import time

class IntroWindow:
    def __init__(self, color='red'):
        # Set color parameters based on choice
        if color.lower() == 'white':
            self.main_color = 'white'
            self.shadow_color = '#404040'  # Dark gray shadow for white text
            self.main_color_value = 255    # FF in hex
            self.shadow_color_value = 64   # 40 in hex
        else:  # default to red
            self.main_color = 'darkred'
            self.shadow_color = '#400000'  # Dark red shadow
            self.main_color_value = 139    # 8B in hex
            self.shadow_color_value = 64   # 40 in hex

        # Create the intro window
        self.root = ctk.CTk()
        self.root.title("Welcome to Pose2Sim")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size (80% of screen size)
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)
        
        # Calculate position for center of screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Set black background
        self.root.configure(fg_color="black")
        
        # Create canvas for animation
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        # Create individual letters with initial opacity
        letters = ['P', 'o', 's', 'e', '2', 'S', 'i', 'm']
        self.text_ids = []
        self.shadow_ids = []  # Add shadow text IDs
        spacing = 50  # Adjust spacing between letters
        total_width = len(letters) * spacing
        start_x = window_width/2 - total_width/2
        
        for i, letter in enumerate(letters):
            # Adjust font size for P and S
            font_size = 78 if letter in ['P', '2', 'S'] else 70
            
            if letter == 'i' or letter == 'm':
                spacing = 49
            elif letter == 'i':
                spacing = 55
            elif letter == 'S':
                spacing = 51
            elif letter == 's':
                spacing = 52
            elif letter == 'o':
                spacing = 54
            # Create shadow text (slightly offset)
            shadow_id = self.canvas.create_text(
                start_x + i * spacing + 2,  # Offset by 2 pixels right
                window_height/2 + 2,        # Offset by 2 pixels down
                text=letter,
                font=('Helvetica', font_size, 'bold'),
                fill=self.shadow_color,
                state='hidden'
            )
            self.shadow_ids.append(shadow_id)
            
            # Create main text
            text_id = self.canvas.create_text(
                start_x + i * spacing,
                window_height/2,
                text=letter,
                font=('Helvetica', font_size, 'bold'),
                fill=self.main_color,
                state='hidden'
            )

            self.text_ids.append(text_id)
            spacing = 50  # Reset spacing for other letters
        
        # Store animation parameters
        self.opacity = 0
        self.fade_step = 0.001
        self.current_group = 0  # Track current group (0: Pose, 1: 2, 2: Sim)
        self.animation_done = False
        self.after_id = None
        
        # Define letter groups (including shadows)
        self.groups = [
            list(zip(self.text_ids[:4], self.shadow_ids[:4])),  # Pose
            list(zip([self.text_ids[4]], [self.shadow_ids[4]])),  # 2
            list(zip(self.text_ids[5:], self.shadow_ids[5:]))   # Sim
        ]
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the fade-in animation after a short delay
        self.after_id = self.root.after(150, self.fade_in)

    def on_closing(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.animation_done = True
        self.root.destroy()

    def fade_in(self):
        if not self.root.winfo_exists():
            return
        if self.current_group < len(self.groups):
            if self.opacity < 1:
                self.opacity += self.fade_step
                # Make current group visible and set opacity
                for text_id, shadow_id in self.groups[self.current_group]:
                    self.canvas.itemconfig(shadow_id, state='normal')
                    self.canvas.itemconfig(text_id, state='normal')
                    main_r = int(self.main_color_value * self.opacity)
                    shadow_r = int(self.shadow_color_value * self.opacity)
                    if self.main_color == 'white':
                        hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                        shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'
                    else:
                        hex_color = f'#{main_r:02x}0000'
                        shadow_color = f'#{shadow_r:02x}0000'
                    self.canvas.itemconfig(shadow_id, fill=shadow_color)
                    self.canvas.itemconfig(text_id, fill=hex_color)
                self.after_id = self.root.after(1, self.fade_in)
            else:
                self.opacity = 0
                self.current_group += 1
                self.after_id = self.root.after(1, self.fade_in)
        else:
            self.opacity = 1
            self.fade_out()

    def fade_out(self):
        if not self.root.winfo_exists():
            return
        if self.opacity > 0:
            self.opacity -= self.fade_step
            # Update all letters opacity together
            main_r = int(self.main_color_value * self.opacity)
            shadow_r = int(self.shadow_color_value * self.opacity)
            if self.main_color == 'white':
                hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'
            else:
                hex_color = f'#{main_r:02x}0000'
                shadow_color = f'#{shadow_r:02x}0000'
            for shadow_id, text_id in zip(self.shadow_ids, self.text_ids):
                self.canvas.itemconfig(shadow_id, fill=shadow_color)
                self.canvas.itemconfig(text_id, fill=hex_color)
            self.after_id = self.root.after(1, self.fade_out)
        else:
            self.animation_done = True
            if self.root.winfo_exists():
                self.on_closing()

    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error during animation: {e}")
            if self.after_id:
                self.root.after_cancel(self.after_id)
        finally:
            self.animation_done = True
        return self.animation_done

if __name__ == "__main__":
    intro = IntroWindow('red')  # or IntroWindow('white')
    intro.run() 
