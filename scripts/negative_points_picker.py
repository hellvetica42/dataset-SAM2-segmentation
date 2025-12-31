from tkinter import Tk, Label, Canvas
from PIL import ImageTk, ImageDraw

MAX_POINTS = 100
negative_points = []
enter_pressed = False

def mouse_click_callback(event, canvas, point_markers):
    global negative_points
    if len(negative_points) < MAX_POINTS:
        print("Negative point on: " + str(event.x) + ", " + str(event.y))
        negative_points.append([event.x, event.y])
        
        # Draw the negative point on canvas
        marker_size = 5
        marker = canvas.create_oval(
            event.x - marker_size, event.y - marker_size,
            event.x + marker_size, event.y + marker_size,
            fill='red', outline='white', width=2
        )
        point_markers.append(marker)
        
        # Close window if max points reached
        if len(negative_points) >= MAX_POINTS:
            event.widget.winfo_toplevel().quit()
    
def enter_callback(event):
    global enter_pressed
    enter_pressed = True
    event.widget.winfo_toplevel().quit()

def choose_negative_points(pil_image, boxes_xyxy=None):
    
    global negative_points, enter_pressed
    
    # Reset state
    negative_points = []
    enter_pressed = False
    
    # Draw boxes on the image
    img_with_boxes = pil_image.copy()
    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        draw = ImageDraw.Draw(img_with_boxes)
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            # Draw box with green color and thicker line
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
    
    root = Tk()
    root.title("Select Negative Points (max 100, press Enter when done)")
    
    # Convert PIL image to PhotoImage
    photo = ImageTk.PhotoImage(img_with_boxes)
    
    # Use Canvas instead of Label to allow drawing
    canvas = Canvas(root, width=img_with_boxes.width, height=img_with_boxes.height)
    canvas.pack()
    
    # Display the image on canvas
    canvas.create_image(0, 0, anchor='nw', image=photo)
    
    # Keep track of point markers
    point_markers = []
    
    # Bind events (pass canvas and point_markers to callback)
    canvas.bind("<Button-1>", lambda e: mouse_click_callback(e, canvas, point_markers))
    root.bind("<Return>", enter_callback)
    
    # Keep reference to avoid garbage collection
    canvas.image = photo
    
    root.mainloop()
    root.destroy()
    
    print(f"Selected {len(negative_points)} negative points")
    return negative_points