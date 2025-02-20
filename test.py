from main_parallel import *
import numpy as np


def test_initialization():
    config = DefaultConfig()
    model = NetworkModel(config)

    # check the uniqueness of node_id and pos
    unique_ids = set(agent.unique_id for agent in model.schedule.agents)
    assert len(unique_ids) == config.num_agents, "Unique ID assignment error"
    positions = set(agent.pos for agent in model.schedule.agents)
    assert len(positions) == config.num_agents, "Position assignment error: duplicate positions found"

    # Check the number of agents
    assert len(model.schedule.agents) == config.num_agents, "Agent count does not match"

    # Check the initial attributes
    for agent in model.schedule.agents:
        assert agent.wealth == config.wealth, "Incorrect initial wealth"
        assert config.U_distribution[0] <= agent.U <= config.U_distribution[1], "U value out of bounds"
        assert np.isclose(np.mean(agent.history), 0), "History not initialized to zeros"
        assert len(agent.history) == 5, "History length is incorrect"
        assert config.eta_distribution[0] - 4 * config.eta_distribution[1] <= agent.eta <= config.eta_distribution[
            0] + 4 * config.eta_distribution[1], "eta out of expected normal distribution range"
        assert config.ration_distribution[0] - 4 * config.ration_distribution[1] <= np.log(agent.ration) <= config.ration_distribution[
            0] + 4 * config.ration_distribution[1], "ration out of expected log normal distribution range"

        # Check belief and rewiring parameters
        assert agent.V_with_belief == 0, "V_with_belief not initialized to 0"
        assert agent.alpha_epsilon == config.alpha_epsilon, "Incorrect alpha_epsilon"
        assert agent.rewiring_prob == config.rewiring_prob, "Incorrect rewiring_prob"
        assert agent.beta_con == config.beta_con, "Incorrect beta_con"
        assert agent.epsilon == config.epsilon, "Incorrect epsilon"

        # Check the number of neighbors
        neighbors_count = len(list(model.G.neighbors(agent.pos)))
        assert neighbors_count >= 0, "Incorrect number of neighbors"

    # Custom values for testing
    test_U = 0.8
    test_eta = 0.2
    test_ration = 1.5
    test_V_with_belief = 0.3
    test_alpha_epsilon = 0.1
    test_rewiring_prob = 0.5
    test_beta_con = 0.7
    test_epsilon = 0.01

    for agent in model.schedule.agents:
        # Assign test values
        agent.U = test_U
        agent.eta = test_eta
        agent.ration = test_ration
        agent.V_with_belief = test_V_with_belief
        agent.alpha_epsilon = test_alpha_epsilon
        agent.rewiring_prob = test_rewiring_prob
        agent.beta_con = test_beta_con
        agent.epsilon = test_epsilon

        # Check if agent attributes are correctly assigned
        assert agent.wealth == config.wealth, "Incorrect initial wealth"
        assert agent.recent_wealth == config.recent_wealth, "Incorrect initial recent wealth"
        assert agent.U == test_U, "U value not correctly assigned"
        assert agent.eta == test_eta, "eta value not correctly assigned"
        assert agent.ration == test_ration, "ration value not correctly assigned"
        assert agent.V_with_belief == test_V_with_belief, "V_with_belief not correctly assigned"
        assert agent.alpha_epsilon == test_alpha_epsilon, "alpha_epsilon not correctly assigned"
        assert agent.rewiring_prob == test_rewiring_prob, "rewiring_prob not correctly assigned"
        assert agent.beta_con == test_beta_con, "beta_con not correctly assigned"
        assert agent.epsilon == test_epsilon, "epsilon not correctly assigned"


    # Check the network nodes and edges
    assert len(model.G.nodes()) == config.num_agents, "The network nodes count does not match"
    assert len(model.G.edges()) >= 0, "Network should have non-negative edges"
    print("Initialization test passed.")


def test_choose_neighbor():
    config = DefaultConfig()
    model = NetworkModel(config)
    agent = model.schedule.agents[0]  # Select an arbitrary agent

    # Ensure all neighbors are marked as not played
    for node_id in model.adjacency[agent.pos]:
        model.id_to_agent[node_id].has_played = False

    chosen_neighbor = agent.choose_neighbor()
    assert chosen_neighbor is None or not chosen_neighbor.has_played, "Chosen neighbor should not have played"

    # Now mark all neighbors as played
    for node_id in model.adjacency[agent.pos]:
        model.id_to_agent[node_id].has_played = True

    chosen_neighbor = agent.choose_neighbor()
    assert chosen_neighbor is None, "If all neighbors have played, should return None"

    print("Choose neighbor test passed.")


def test_add_belief():
    config = DefaultConfig()
    model = NetworkModel(config)

    agent0 = Player(0, model, 0, config)
    agent1 = Player(1, model, 1, config)

    # Test normal behavior
    agent0.add_belief(agent1)

    expected_U_with_belief = agent0.U - np.tanh(agent0.alpha_epsilon * (agent0.U - agent1.U)) * agent0.U
    expected_V_with_belief = agent0.V - np.tanh(agent0.alpha_epsilon * (agent0.V - agent1.V)) * agent0.V

    assert np.isclose(agent0.U_with_belief,
                      expected_U_with_belief), f"U_with_belief incorrect: {agent0.U_with_belief} != {expected_U_with_belief}"
    assert np.isclose(agent0.V_with_belief,
                      expected_V_with_belief), f"V_with_belief incorrect: {agent0.V_with_belief} != {expected_V_with_belief}"

    # the other agent unchanged
    assert agent1.U_with_belief == 0, "agent1.U_with_belief should not change"
    assert agent1.V_with_belief == 0, "agent1.V_with_belief should not change"

    # Test alpha_epsilon = 0 case
    agent0.alpha_epsilon = 0
    agent0.add_belief(agent1)
    assert agent0.U_with_belief == agent0.U, "U_with_belief should equal U when alpha_epsilon is 0"
    assert agent0.V_with_belief == agent0.V, "V_with_belief should equal V when alpha_epsilon is 0"

    # Test large alpha_epsilon case
    agent0.U = 10.0
    agent1.U = 0.1
    agent0.V = 10.0
    agent1.V = 0.1
    agent0.alpha_epsilon = 100
    agent0.add_belief(agent1)
    assert abs(agent0.U_with_belief) < 0.01, "U_with_belief should approach 0 for large alpha_epsilon"
    assert abs(agent0.V_with_belief) < 0.01, "V_with_belief should approach 0 for large alpha_epsilon"

    print("add_belief test passed.")


def test_choose_strategy():

    config = DefaultConfig()

    # test1: rational players choose (C,C) for [[1,0.5], [0.5,0]]
    agent1 = Player(1, None, None, config, ration=100)
    agent2 = Player(2, None, None, config, ration=100)

    agent1.U_with_belief = 0.5
    agent1.V_with_belief = 0.5
    agent2.U_with_belief = 0.5
    agent2.V_with_belief = 0.5

    E1, E2 = agent1.choose_strategy(agent2)

    assert E1 == 1, f"Strategy for agent1 incorrect"
    assert E2 == 1, f"Strategy for agent2 incorrect"


    # test 2: rational player E1 choose C for [[1,1], [0,0]]
    agent2 = Player(2, None, None, config, ration=0)

    agent1.U_with_belief = 1
    agent1.V_with_belief = 0

    E1, _ = agent1.choose_strategy(agent2)

    assert E1 == 1, f"Strategy for agent1 incorrect"

    print("Strategy choice test passed.")


def test_choose_strategy2():
    # test the random choice scenario for player with ration=0
    config = DefaultConfig()

    agent1 = Player(1, None, None, config, ration=100)
    agent2 = Player(2, None, None, config, ration=0)

    agent1.U_with_belief = 1
    agent1.V_with_belief = 0
    agent2.U_with_belief = 0
    agent2.V_with_belief = 0.5

    count1 = 0
    count0 = 0

    for i in range(100):
        _, E2 = agent1.choose_strategy(agent2)
        if E2 == 1:
            count1 += 1
        elif E2 == 0.5:
            count0 += 1

    # count1 and count0 should be roughly the same
    assert abs(count1 - count0) <= 10, "The counts are not roughly the same."
    # print("The number of times E1 equal to 1 is", count1)
    # print("The number of times E1 equal to 0 is", count0)

    assert count1+count0 == 100, "Strategy for agent2 incorrect"

    print("Strategy choice test passed.")


def test_agent_interaction():
    config = DefaultConfig()
    model = NetworkModel(config)
    initial_step_count = model.steps
    initial_eta = {agent.unique_id: agent.eta for agent in model.schedule.agents}
    initial_ration = {agent.unique_id: agent.ration for agent in model.schedule.agents}

    model.step()

    # Check if step() increments
    assert model.steps == initial_step_count + 1, "model.steps does not increase"

    # Check at least one agent's has_played changed
    assert any(agent.has_played for agent in model.schedule.agents), "No agent played during the step"

    # Check fixed eta and ration
    assert all(agent.eta == initial_eta[agent.unique_id] for agent in
               model.schedule.agents), "Agent eta should not be changed"
    assert all(agent.ration == initial_ration[agent.unique_id] for agent in
               model.schedule.agents), "Agent ration did not changed"

    print("Agent interaction test passed.")


def test_wealth_update():
    config = DefaultConfig()
    model = NetworkModel(config)
    initial_wealths = {agent.unique_id: agent.wealth for agent in model.schedule.agents}
    initial_recent_wealths = {agent.unique_id: agent.recent_wealth for agent in model.schedule.agents}
    initial_histories = {agent.unique_id: agent.history.copy() for agent in model.schedule.agents}

    model.step()

    # Check wealth changes
    assert any(agent.wealth != initial_wealths[agent.unique_id] for agent in
               model.schedule.agents), "Agent wealth did not change as expected"
    assert any(agent.recent_wealth != initial_recent_wealths[agent.unique_id] for agent in
               model.schedule.agents), "Agent recent wealth did not change as expected"

    # Check for history update
    assert any(not np.array_equal(agent.history, initial_histories[agent.unique_id]) for agent in
               model.schedule.agents), "Agent history did not update as expected"
    for agent in model.schedule.agents:
        assert np.array_equal(agent.history[:-1],
                              initial_histories[agent.unique_id][1:]), "History did not shift correctly"

    print("Agent update test passed.")


def test_uv_update():
    config = DefaultConfig()

    agent1 = Player(1, None, None, config)
    agent2 = Player(2, None, None, config)

    agent1.U = 0.5
    agent1.V = 0.5
    agent2.U = 1
    agent2.V = 1

    agent1.wealth = 0
    agent1.recent_wealth = 0

    agent2.wealth = 100
    agent2.recent_wealth = 100

    agent1.update_payoff_mx(agent2)

    assert (agent1.U != 0.5 or agent1.V != 0.5), "Agent 0's U, V should have been updated"
    assert (agent2.U == 1 and agent2.V == 1), "Agent 1's U, V should not have been updated"


def test_uv_update_advanced():

    class SmallConfig(DefaultConfig):
        def __init__(self):
            super().__init__()
            self.num_agents = 2
            self.network_type = 'regular'
            self.d = 1
            self.rewiring_prob = 0

    config = SmallConfig()
    model = NetworkModel(config)

    agent1 = model.schedule.agents[0]
    agent2 = model.schedule.agents[1]
    agent1.wealth = 0
    agent2.wealth = 100
    agent1.recent_wealth = 0
    agent2.recent_wealth = 100

    old_U0, old_V0 = agent1.U, agent1.V
    old_U1, old_V1 = agent2.U, agent2.V

    agent1.update_payoff_mx(agent2)

    assert (agent1.U != old_U0 or agent1.V != old_V0), "Agent 0's U, V should have been updated"
    assert (agent2.U == old_U1 and agent2.V == old_V1), "Agent 1's U, V should not have been updated"
    print("UV update test passed.")


def test_rewiring():
    config = DefaultConfig()

    # rewiring prob = 1
    config.rewiring_prob = 1.0
    model = NetworkModel(config)
    initial_edges = sum(len(model.adjacency[n]) for n in model.adjacency) // 2

    model.step()
    after_edges = sum(len(model.adjacency[n]) for n in model.adjacency) // 2

    assert initial_edges != after_edges, "Edges didn't change when rewiring prob=1."

    # rewiring prob = 0
    config.rewiring_prob = 0.0
    model = NetworkModel(config)
    initial_edges = sum(len(model.adjacency[n]) for n in model.adjacency) // 2

    model.step()
    after_edges = sum(len(model.adjacency[n]) for n in model.adjacency) // 2

    assert initial_edges == after_edges, "Edges changed when rewiring prob=0."


    print("Rewiring test passed.")


def test_calculators():
    # Testing compute_gini
    assert NetworkModel.compute_gini([]) == 0.0, "compute_gini failed"
    assert NetworkModel.compute_gini([0, 0, 0]) == 0.0, "compute_gini failed"
    assert NetworkModel.compute_gini([1, 1, 1, 1]) == 0.0, "compute_gini failed"
    assert round(NetworkModel.compute_gini([1, 2, 3, 4]), 2) == 0.25, "compute_gini failed"

    # Testing compute_average_path_length
    G_complete = nx.complete_graph(4)
    assert NetworkModel.compute_average_path_length(G_complete) == 1, "compute_average_path_length failed"
    G_disconnected = nx.Graph([(0, 1), (2, 3)])
    assert NetworkModel.compute_average_path_length(G_disconnected) == 1.0, "compute_average_path_length failed"

    # Testing avg_degree
    adjacency = {
        0: {1, 3},
        1: {0, 2},
        2: {1, 3},
        3: {0, 2}
    }
    # the name of class 'MockModel'; bases=(), no parent class; attribute dictionary {'adjacency': adjacency}
    # () instantiate the class MockModel
    mock_model = type('MockModel', (), {'adjacency': adjacency})()
    val = NetworkModel.avg_degree(mock_model)
    assert val == 2, f"Expected 2, got {val}"

    adjacency = {
        0: {},
        1: {},
        2: {}
    }
    mock_model = type('MockModel', (), {'adjacency': adjacency})()
    val = NetworkModel.avg_degree(mock_model)
    assert val == 0, f"Expected 0, got {val}"

    print("Calculators test passed.")


def test_data_collection():
    config = DefaultConfig()
    model = NetworkModel(config)
    steps_to_run = 20
    model.run_model(steps_to_run)

    # Check model_reporters
    model_data = model.datacollector.get_model_vars_dataframe()
    assert len(model_data) == 2, "Incorrect data count"
    required_cols = ["Gini", "Average Degree", "Clustering Coefficient", "Average Path Length", "Wealth Distribution"]
    for col in required_cols:
        assert col in model_data.columns, f"{col} is not in the data"

    # Check agent_reporters
    agent_data = model.datacollector.get_agent_vars_dataframe()
    required_agent_cols = ["Wealth", "Recent Wealth", "Risk Aversion", "Rationality", "U", "V"]
    for col in required_agent_cols:
        assert col in agent_data.columns, f"{col} is not in the data"

    print("Data collection test passed.")


if __name__ == "__main__":
    test_initialization()
    test_choose_neighbor()
    test_add_belief()
    test_choose_strategy()
    test_choose_strategy2()
    test_agent_interaction()
    test_wealth_update()
    test_uv_update()
    test_uv_update_advanced()
    test_rewiring()
    test_calculators()
    test_data_collection()