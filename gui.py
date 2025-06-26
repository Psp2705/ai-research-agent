import tkinter as tk
from tkinter import scrolledtext, messagebox
from agent import run_agent  # <-- We'll create this function in agent.py

def run_query():
    query = query_entry.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a research query.")
        return

    output_text.delete('1.0', tk.END)  # Clear previous output
    try:
        structured, filename, tool_outputs = run_agent(query)
        output_text.insert(tk.END, f"📄 Topic: {structured.topic}\n\n")
        output_text.insert(tk.END, f"{structured.summary}\n\n")
        output_text.insert(tk.END, f"Sources:\n- " + "\n- ".join(structured.sources) + "\n\n")
        output_text.insert(tk.END, f"Tools Used: {', '.join(structured.tools_used)}\n")
        output_text.insert(tk.END, f"\n✅ Saved to: {filename}")
    except Exception as e:
        messagebox.showerror("Agent Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("🔍 AI Research Assistant (Gemini)")
root.geometry("700x500")

tk.Label(root, text="Enter your research query:", font=("Arial", 12)).pack(pady=10)
query_entry = tk.Entry(root, width=80)
query_entry.pack()

tk.Button(root, text="Run Agent", command=run_query, bg="#4CAF50", fg="white").pack(pady=10)

output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=85, height=20)
output_text.pack(pady=10)

root.mainloop()
