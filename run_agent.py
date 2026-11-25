from src.orchestrator import (
    count_tool_results,
    open_agent,
    resume_approval,
    start_investigation,
)

DEFAULT_QUESTION = """
Investigate shipment record_id 1. Retrieve its complete telemetry from MySQL,
search applicable cold-chain SOP rules, and fetch current weather using the
shipment's actual GPS coordinates. Assess temperature, delay probability,
risk classification, port congestion, and route risk index. Distinguish
historical shipment data from live environmental conditions. Identify missing
evidence, evaluate the evidence, and use additional tools only when they can
materially improve the investigation. Do not execute any real operational action.
"""


def main():
    app, checkpoint_connection = open_agent()
    try:
        print("\n========================================")
        print("       COLDCHAIN AI INVESTIGATION")
        print("========================================")

        question = input(
            "\nEnter an investigation request "
            "(press Enter for record_id 1): "
        ).strip() or DEFAULT_QUESTION

        result, config, thread_id = start_investigation(app, question)

        print("\n========================================")
        print("           INVESTIGATION REPORT")
        print("========================================\n")
        print(result.get("investigation_report") or "No report returned.")

        interrupts = result.get("__interrupt__", [])
        if interrupts:
            request = interrupts[0].value
            print("\n========================================")
            print("             ACTION APPROVAL")
            print("========================================")
            print(request.get("message", "Approval required."))
            print("\nProposed action:")
            print(request.get("action", {}))

            answer = input("\nApprove simulated action? (yes/no): ").strip().lower()
            approved = answer in {"yes", "y"}
            result = resume_approval(
                app,
                config,
                approved=approved,
                approver="local_operator",
            )
        else:
            print("\nNo approval interrupt was returned. No action was approved.")

        print("\n========================================")
        print("              ACTION RESULT")
        print("========================================\n")
        print(result.get("action_result", "No action result returned."))
        print("\nTool results:", count_tool_results(result.get("messages", [])))
        print("Replanning rounds:", result.get("replan_round", 0))
        print("Thread ID:", thread_id)
        print("Audit log: logs/action_audit.jsonl")
        print("Checkpoint DB: data/coldchain_checkpoints.sqlite")
    finally:
        checkpoint_connection.close()


if __name__ == "__main__":
    main()
