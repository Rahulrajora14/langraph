history = list(
    workflow.get_state_history(config)
)

checkpoint_id = history[2].config["configurable"]["checkpoint_id"]

past_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": checkpoint_id
    }
}

workflow.invoke(
    None,
    config=past_config
)