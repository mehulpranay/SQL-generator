
# 🚀 LLM-SQL: 8B Model vs. Spider Benchmark

### **The Project**
SQL generation seems solved until you realize GPT-3.5 zero-shot hits only **70% on Spider**. I matched that score using an **8B model** on a free Kaggle GPU by proving one non-obvious bet: **the schema representation matters more than the model.**

---

### **Architecture & Data Flow**

The system prioritizes high-fidelity schema mapping to bridge the gap between natural language and database constraints.



**Sample Input/Output:**
* **NL Input:** *"Which stadiums have a capacity higher than the average?"*
* **Model Output:** ```sql
    SELECT name FROM stadium WHERE capacity > (SELECT AVG(capacity) FROM stadium)
    ```

---

### **Quick Start (Local)**

Run the inference engine locally in three simple steps:

1. **Clone & Install**
   ```bash
   git clone [https://github.com/user/llm-sql](https://github.com/user/llm-sql) && cd llm-sql && pip install -r requirements.txt
   
