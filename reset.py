import os
import tkinter as tk
from tkinter import messagebox


FILE_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\emotion_model_realworld.h5"
FILE_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\emotion_model_realworld2.h5"


def check_file():
    """Check if the file exists and update label."""
    if os.path.exists(FILE_PATH):
        status_label.config(text="File exists ✅", fg="green")
    else:
        status_label.config(text="File does NOT exist ❌", fg="red")

def delete_file():
    """Delete the file if it exists and user confirms."""
    if not os.path.exists(FILE_PATH):
        messagebox.showinfo("Delete File", "File does not exist. Nothing to delete.")
        check_file()
        return

    # Ask user for confirmation
    answer = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete:\n{FILE_PATH}?")
    if answer:
        try:
            os.remove(FILE_PATH)
            messagebox.showinfo("Delete File", "File deleted successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete file.\nError: {e}")
    # Update status after attempt
    check_file()

# =========================
# GUI setup
# =========================
root = tk.Tk()
root.title("Emotion Model File Checker")

# Window size and position
root.geometry("550x200")

# File path label
path_label = tk.Label(root, text=f"File Path:\n{FILE_PATH}", wraplength=500, justify="left")
path_label.pack(pady=10)

# Status label
status_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
status_label.pack(pady=5)

# Buttons frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

check_btn = tk.Button(btn_frame, text="Check File", width=15, command=check_file)
check_btn.grid(row=0, column=0, padx=10)

delete_btn = tk.Button(btn_frame, text="Delete File", width=15, command=delete_file)
delete_btn.grid(row=0, column=1, padx=10)

# Run initial check on startup
check_file()

root.mainloop()
