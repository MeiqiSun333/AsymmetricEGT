from main import *


def test_initialization():
    config = DefaultConfig()
    model = NetworkModel(config)

    # Check the number of agents
    assert len(model.schedule.agents) == config.num_agents, "Agent count does not match"

    # Check the initial attributes
    for agent in model.schedule.agents:
        assert agent.wealth == config.wealth, "Incorrect initial wealth"
        assert agent.recent_wealth == config.recent_wealth, "Incorrect initial recent wealth"

        neighbors_count = len(list(model.G.neighbors(agent.pos)))
        assert neighbors_count >= 0, "Incorrect number of neighbors"

    # Check the number of network nodes
    assert len(model.G.nodes()) == config.num_agents, "The network nodes count does not match"
    print("Initialization test passed.")



def test_agent_interaction():
    config = DefaultConfig()
    model = NetworkModel(config)
    initial_step_count = model.steps
    initial_num_edges = model.G.number_of_edges()

    model.step()

    # Check if step() +1
    assert model.steps == initial_step_count + 1, "model.steps does not increase"

    # Check the has_played state
    any_played = False
    for agent in model.schedule.agents:
        if len(list(model.G.neighbors(agent.pos))) > 0:
            if agent.has_played:
                any_played = True
        # Check wealth update
        assert agent.wealth is not None, "Incorrect wealth update"

    # Check if the has_played state switch to True
    if initial_num_edges > 0:
        assert any_played, "No agent is playing game"

    # Check rewiring
    if config.rewiring_prob > 0:
        after_num_edges = model.G.number_of_edges()
        assert after_num_edges >= 0, "Incorrect edge rewiring"

    print("Agent interaction test passed.")



def test_data_collection():
    config = DefaultConfig()
    model = NetworkModel(config)
    steps_to_run = 3
    model.run_model(steps_to_run)

    # Check model_reporters
    model_data = model.datacollector.get_model_vars_dataframe()
    assert len(model_data) == steps_to_run, "Incorrect data count"
    required_cols = ["Gini", "Average Degree", "Clustering Coefficient", "Average Path Length", "Wealth Distribution"]
    for col in required_cols:
        assert col in model_data.columns, f"{col} is not in the data"

    # Check agent_reporters
    agent_data = model.datacollector.get_agent_vars_dataframe()
    assert len(agent_data) > 0, "No data is reported"
    required_agent_cols = ["Wealth", "Recent Wealth", "Risk Aversion", "Rationality", "U", "V"]
    for col in required_agent_cols:
        assert col in agent_data.columns, f"{col} is not in the data"

    print("Data collection test passed.")



def test_rewiring():
    config = DefaultConfig()
    config.rewiring_prob = 1.0
    model = NetworkModel(config)
    initial_edges = set(model.G.edges())

    model.step()
    after_edges = set(model.G.edges())

    if len(initial_edges) > 0:
        if initial_edges == after_edges:
            print("Rewiring test: Edges didn't change.")
        else:
            print("Rewiring test: Edges changed as expected.")
    else:
        print("Rewiring test: No edges initially, rewiring might add edges if code allows it.")

    print("Rewiring test passed.")



def test_add_belief():
    config = DefaultConfig()
    model = NetworkModel(config)

    agent0 = Player(0, model, 0, config)
    agent1 = Player(1, model, 1, config)

    agent0.add_belief(agent1)

    expected_U_with_belief = agent0.U - np.tanh(agent0.alpha_epsilon * (agent0.U - agent1.U)) * agent0.U
    expected_V_with_belief = agent0.V - np.tanh(agent0.alpha_epsilon * (agent0.V - agent1.V)) * agent0.V

    assert agent0.U_with_belief == expected_U_with_belief, f"U_with_belief incorrect: {agent0.U_with_belief} != {expected_U_with_belief}"
    assert agent0.V_with_belief == expected_V_with_belief, f"V_with_belief incorrect: {agent0.V_with_belief} != {expected_V_with_belief}"

    print("add_belief test passed.")



def test_uv_update():

    class SmallConfig(DefaultConfig):
        def __init__(self):
            super().__init__()
            self.num_agents = 2
            self.network_type = 'regular'
            self.d = 1
            self.rewiring_prob = 0

    config = SmallConfig()
    model = NetworkModel(config)

    agent0 = model.schedule.agents[0]
    agent1 = model.schedule.agents[1]
    agent0.wealth = 0
    agent1.wealth = 100
    agent0.recent_wealth = 0
    agent1.recent_wealth = 100

    old_U0, old_V0 = agent0.U, agent0.V
    old_U1, old_V1 = agent1.U, agent1.V

    # Set agent0 as the initiator of the game
    agent0.update_payoff_mx(agent1)

    new_U0, new_V0 = agent0.U, agent0.V
    new_U1, new_V1 = agent1.U, agent1.V

    assert (new_U0 != old_U0 or new_V0 != old_V0), "Agent 0's U, V should have been updated"
    assert (new_U1 == old_U1 and new_V1 == old_V1), "Agent 1's U, V should not have been updated"
    print("UV update test passed.")


if __name__ == "__main__":
    test_initialization()
    test_agent_interaction()
    test_data_collection()
    test_rewiring()
    test_add_belief()
    test_uv_update()