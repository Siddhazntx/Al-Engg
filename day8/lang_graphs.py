from typing import TypedDict
from langgraph.graph import END, StateGraph

class State(TypedDict):
    number : int

def double_num(state:State) ->dict:
    board_num = state["number"]
    new_num = board_num*2
    print("in double")
    print(new_num) 

    return {"number":new_num}

def finish(state:State) ->dict:
    board_num = state["number"]
    
    print("finish")
    print(board_num) 

    return {"number":board_num}

def decision(state:State)->str:
    if state["number"] < 100:
        return "double"
    else:
        return "finish"

builder = StateGraph(State)
builder.add_node("double_num",double_num)
builder.add_node("finish",finish)

builder.set_entry_point("double_num")
builder.add_conditional_edges(
    "double_num",
    decision,
    {"double":"double_num", "finish":"finish"}
)

builder.add_edge("finish",END)
graph = builder.compile()

if __name__ == '__main__':
    result = graph.invoke({"number":5})
    print("Final result:", result)

