import numpy as np
import pickle
import os
import networkx as nx
import symengine as sym

import sys
from optimizers.adam import *
import time
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import enum
from typing import Optional, List, Tuple, Callable
from pathlib import Path
import pandas as pd

np.set_printoptions(suppress=True)


# np.set_printoptions(threshold=sys.maxsize)

def random_weights(graph: nx.Graph,
                   rs: Optional[np.random.RandomState] = None,
                   type: str = 'bimodal'):
    """Take a graph and make an equivalent graph with weights of plus or minus
    one on each edge.
    Args:
    graph: A graph to add weights to
    rs: A RandomState for making replicable experiments. If not provided,
        the global numpy random state will be used. Please be careful
        when generating random problems. You should construct *one*
        seeded RandomState at the beginning of your script and use
        that one RandomState (in a deterministic fashion) to generate
        all the problem instances you may need.
    """

    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
    return problem_graph


def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')


class Graph:

    def __init__(self, n, d=3, G=None):
        self.n = n
        self.d = d
        if G is None:
            G = nx.generators.random_graphs.random_regular_graph(d, n)
            G = random_weights(graph=G, rs=np.random.RandomState(42))
            for e in G.edges():
                G[e[0]][e[1]]['weight'] = 2 * np.random.randint(2) - 1
        self.G = G
        self.G0 = G.copy()

    def reset(self):
        self.G = self.G0.copy()

    def get_G_numpy(self, nodelist=None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_numpy_array(self.G, dtype=np.float64, nodelist=nodelist)

    def get_G_sparse(self, nodelist=None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_scipy_sparse_matrix(self.G, dtype=np.float64, nodelist=nodelist)

    def eliminate(self, edge, sign):
        rmv_edges = []
        updt_edges = []
        add_edges = []
        for neighb in self.G.neighbors(edge[1]):
            rmv_edges += [(edge[1], neighb)]
            if neighb not in edge:
                if (edge[0], neighb) in self.G.edges():
                    self.G[edge[0]][neighb]['weight'] += sign * self.G[edge[1]][neighb]['weight']
                    if self.G[edge[0]][neighb]['weight'] == 0:
                        self.G.remove_edge(edge[0], neighb)
                        rmv_edges += [(edge[0], neighb)]
                    else:
                        updt_edges += [(edge[0], neighb)]
                else:
                    self.G.add_edge(edge[0], neighb, weight=sign * self.G[edge[1]][neighb]['weight'])
                    add_edges += [(edge[0], neighb)]
        for e in self.G.edges(edge[0]):
            if e not in updt_edges + rmv_edges + add_edges:
                updt_edges += [e]
        for neighb in self.G.neighbors(edge[1]):
            for e in self.G.edges(neighb):
                if e not in updt_edges + rmv_edges + add_edges:
                    updt_edges += [e]
        # cst = sign * self.G[edge[0]][edge[1]]['weight']
        self.G.remove_node(edge[1])
        return rmv_edges, updt_edges, add_edges


class RQAOA_agent:

    def __init__(self, n, nc, G, d, learning_rates, gamma, init_beta, init_angles, batch_size, normalize, reward_fct,
                 pickle_path=None, run='0', idx='0', G_name=''):
        self.n = n
        self.nc = nc
        self.G = G
        self.graph = Graph(n, d, G)
        self.G_name = G_name
        self.learning_rates = learning_rates
        self.grid_N = 1000
        self.gamma = gamma
        self.init_beta = init_beta
        self.init_angles = init_angles
        self.batch_size = batch_size
        self.normalize = normalize
        self.reward_fct = reward_fct
        self.x = sym.Symbol('x', real=True)
        self.y = sym.Symbol('y', real=True)
        if init_beta[0] == "all":
            self.betas = init_beta[1] * np.ones(n - nc)
        elif init_beta[0] == "one":
            self.betas = np.array([init_beta[1]], dtype=float)
        elif init_beta[0] == "one-all":
            self.betas = init_beta[1] * np.ones(int((n ** 2 - n) / 2))
        elif init_beta[0] == "all-all":
            self.betas = init_beta[1] * np.ones((n - nc) * int((n ** 2 - n) / 2))

        if init_angles[0] == "rqaoa":
            if init_angles[2] is None:
                batch_rqaoa_angles, batch_energies, batch_zs = [], [], []
                for i in range(init_angles[1]):
                    print("RQAOA agent " + str(i + 1) + "/" + str(init_angles[1]))
                    rqaoa_angles, energy, z = self.rqaoa()
                    batch_rqaoa_angles += [rqaoa_angles]
                    batch_energies += [energy]
                    batch_zs += [z]
                    print("Energy: " + str(energy))
                arg = np.argmax(batch_energies)
                rqaoa_angles = np.array(batch_rqaoa_angles[arg], dtype=float)
                ref = (batch_energies[arg], batch_zs[arg], batch_energies)
            else:
                rqaoa_angles, ref = pickle.load(open(init_angles[2], 'rb'))
            self.all_angles = rqaoa_angles
            self.ref = ref
        elif init_angles[0] == "zero":
            self.all_angles = np.random.randn(2 * (n - nc)) * init_angles[1]
            self.ref = None
        elif init_angles[0] == "xtrm":
            self.all_angles = np.array([0.21991149, 0.39443473] * (n - nc))
            self.ref = None

        self.optimizer = AdamOptimizer([self.all_angles, self.betas], learning_rate_init=learning_rates, amsgrad=True)

        self.history_rewards = []

        self.run = run
        if pickle_path is None:
            pickle_path = './trained_agents/run' + str(self.run) + '/rqaoa_agent' + str(self.n) + '-' + str(
                self.nc) + '_' + self.G_name + '_lr' + str(self.learning_rates[0]) + '-' + str(
                self.learning_rates[1]) + '_gamma' + str(self.gamma) + '_rwd-' + str(
                self.reward_fct) + '_batchsize' + str(self.batch_size) + '_beta-' + init_beta[0] + '-' + str(
                init_beta[1]) + '_ang-' + init_angles[0] + '-' + str(
                init_angles[1]) + self.normalize * '_norm' + '_' + str(idx) + '.pckl'
        self.pickle_path = pickle_path

    def play_train(self, nb_batches, train=True, store=True):
        self.graph.reset()

        nodelist = np.array(sorted(self.G.nodes()))
        J = self.graph.get_G_numpy(nodelist)

        f_s0, _, action_space0 = self.generate_fs_h_actions(J, nodelist)

        for batch in range(nb_batches):
            batch_grad_xy = []
            batch_grad_beta = []
            batch_returns = []
            for episode in range(self.batch_size):
                self.graph.reset()
                nodelist = np.arange(self.n)
                f_s, action_space = f_s0.copy(), action_space0.copy()
                assignments, signs = [], []
                grad_xy, grad_beta = [], []
                for m in range(self.n - self.nc):
                    angles = self.all_angles[2 * m:2 * m + 2]
                    expectations, indcs = self.compute_expectations(f_s, angles)
                    abs_expectations = np.abs(expectations)
                    betas_idx, betas, beta = [], [], 1.
                    if self.init_beta[0] in ["one-all", "all-all"]:
                        betas_idx = [self.get_idx_beta(e) for e in action_space]
                        betas = [self.betas[i] for i in betas_idx]
                        policy = self.softmax(betas * abs_expectations, 1.)
                    else:
                        if self.init_beta[0] == "one":
                            beta = self.betas[0]
                        else:
                            beta = self.betas[m]
                        policy = self.softmax(abs_expectations, beta)
                    idx = np.random.choice(range(len(policy)), p=policy)
                    edge, sign = action_space[idx], np.sign(expectations[idx])

                    # Compute step grads
                    if train:
                        if self.init_beta[0] in ["one-all", "all-all"]:
                            if self.learning_rates[0] > 0:

                                grad_xy += self.compute_log_pol_diff(f_s, indcs, policy, expectations, betas, idx, sign,
                                                                     angles)
                            grad = np.zeros(int((self.n ** 2 - self.n) / 2))
                            grad[betas_idx[idx]] += abs_expectations[idx]
                            for i in range(len(action_space)):
                                grad[betas_idx[i]] -= policy[i] * abs_expectations[i]
                            grad_beta += [grad]
                        else:
                            if self.learning_rates[0] > 0:
                                betas = np.ones(len(f_s)) * beta
                                
                                grad_xy += self.compute_log_pol_diff(f_s, indcs, policy, expectations, betas, idx, sign,
                                                                     angles)
                            grad_beta += [abs_expectations[idx] - np.dot(policy, abs_expectations)]

                    # Update graph and assignments
                    rmv_edges, updt_edges, add_edges = self.graph.eliminate(edge, sign)
                    assignments += [edge]
                    signs += [sign]
                    nodelist = nodelist[nodelist != edge[1]]
                    if m < self.n - self.nc - 1:
                        J = self.graph.get_G_numpy(nodelist)
                        f_s, action_space = self.update(f_s, action_space, J, nodelist, rmv_edges, updt_edges,
                                                        add_edges)
                # Brute force
                J = self.graph.get_G_numpy(nodelist)
                _, z_c = self.bruteforce(J, self.nc)

                # Outcome
                z_s, ep_energies, ep_contribs = self.expand_result(z_c, assignments, signs, nodelist)
                self.history_rewards += [ep_energies[0]]
                print(f'Rewards: {ep_energies[0]}')
                # Compute returns
                if self.reward_fct == "nrg":
                    disct_reward = ep_energies[0]
                    returns = [disct_reward]
                    for i in range(1, self.n - self.nc):
                        disct_reward *= self.gamma
                        returns.insert(0, disct_reward)
                elif self.reward_fct == "ep-cntrb":
                    returns = []
                    discounted_sum = 0
                    for r in ep_contribs[::-1]:
                        discounted_sum = r + self.gamma * discounted_sum
                        returns.insert(0, discounted_sum)
                else:
                    returns = ep_energies
                batch_returns += [returns]

                # Add episode contrib to batch_grad
                if train:
                    batch_grad_xy += [grad_xy]
                    batch_grad_beta += [grad_beta]

            if train:
                batch_returns = np.array(batch_returns)
                if self.normalize:
                    batch_returns = batch_returns - np.mean(batch_returns, axis=0)

                avg_grad_xy = np.zeros_like(self.all_angles)
                avg_grad_beta = np.zeros_like(self.betas)
                for i in range(self.batch_size):
                    returns = batch_returns[i]
                    grad_xy = batch_grad_xy[i]
                    grad_beta = batch_grad_beta[i]
                    for j in range(self.n - self.nc):
                        avg_grad_xy[2 * j] += returns[j] * grad_xy[2 * j]
                        avg_grad_xy[2 * j + 1] += returns[j] * grad_xy[2 * j + 1]
                    if self.init_beta[0] == "one":
                        avg_grad_beta += np.dot(np.array(returns), grad_beta)
                    elif self.init_beta[0] == "one-all":
                        avg_grad_beta += np.sum([returns[j] * grad_beta[j] for j in range(len(returns))], axis=0)
                    elif self.init_beta[0] == "all-all":
                        avg_grad_beta += np.concatenate([returns[j] * grad_beta[j] for j in range(len(returns))],
                                                        axis=0)
                    else:
                        avg_grad_beta += np.array(returns) * grad_beta
                avg_grad_xy /= self.batch_size
                avg_grad_beta /= self.batch_size

                # Update params
                updates = self.optimizer.get_updates([avg_grad_xy, avg_grad_beta])
                self.all_angles += updates[0]
                self.betas += updates[1]

            if store:
                self.store_agent()

    def rqaoa(self):

        self.graph.reset()
        nodelist = np.array(sorted(self.G.nodes()))
        J = self.graph.get_G_numpy(nodelist)

        f_s, h, action_space = self.generate_fs_h_actions(J, nodelist)
        assignments = []
        signs = []
        rqaoa_angles = np.array([], dtype=float)
        ts = time.time()
        for m in range(self.n - self.nc):
            angles, f_val = self.compute_extrema(h)
            # angles = solutions[np.argmax(extrema)]
            rqaoa_angles = np.append(rqaoa_angles, angles)
            expectations, indcs = self.compute_expectations(f_s, angles)
            abs_expectations = np.abs(expectations)
            max_abs = np.flatnonzero(abs_expectations == abs_expectations.max())
            idx = np.random.choice(max_abs)
            edge, sign = action_space[idx], np.sign(expectations[idx])
            rmv_edges, updt_edges, add_edges = self.graph.eliminate(edge, sign)
            assignments += [edge]
            signs += [sign]
            nodelist = nodelist[nodelist != edge[1]]
            J = self.graph.get_G_numpy(nodelist)
            f_s, action_space = self.update(f_s, action_space, J, nodelist, rmv_edges, updt_edges, add_edges)
            h = self.compute_h(f_s, action_space, J, nodelist)
        _, z_c = self.bruteforce(J, self.nc)
        z_s, ep_energies, ep_contribs = self.expand_result(z_c, assignments, signs, nodelist)

        return rqaoa_angles, ep_energies[0], z_s[0]

    def update(self, f_s, action_space, J, nodelist, rmv_edges, updt_edges, add_edges):
        nl = list(nodelist)
        for edge in rmv_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            indx = action_space.index(edge)
            action_space.pop(indx)
            f_s.pop(indx)

        for edge in add_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            inserted = False
            for i in range(len(action_space)):
                edge_i = action_space[i]
                if (edge_i[0] == edge[0] and edge_i[1] > edge[1]) or edge_i[0] > edge[0]:
                    action_space.insert(i, edge)
                    f_s.insert(i, self.compute_f(J, nl.index(edge[0]), nl.index(edge[1])))
                    inserted = True
                    break
            if not inserted:
                action_space += [edge]
                f_s += [self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))]

        for edge in updt_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            if edge in action_space:
                indx = action_space.index(edge)
                f_s[indx] = self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))

        return f_s, action_space

    def compute_h(self, f_s, action_space, J, nodelist):
        h = 0.
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        print("Wrong count")
                    h += J[i, j] * f_s[count]
                    count += 1
        return h

    def compute_h_diff(self, f_s, indcs, action_space, J, nodelist):
        x = self.x
        y = self.y
        gather = np.zeros_like(f_s)
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        print("Wrong count")
                    gather[indcs[count]] += J[i, j]
                    count += 1
        diff_h_x = 0.
        diff_h_y = 0.
        for i in range(len(gather)):
            if gather[i]:
                diff_h_x += gather[i] * f_s[i].diff(x)
                diff_h_y += gather[i] * f_s[i].diff(y)
        return diff_h_x, diff_h_y

    def compute_expectations(self, f_s, angles):
        x = self.x
        y = self.y
        expectations = []
        indcs = []
        for i, f in enumerate(f_s):
            if f in f_s[:i]:
                indx = f_s.index(f)
                expectations += [expectations[indx]]
                indcs += [indx]
            else:
                expectations += [float(f.subs({x: angles[0], y: angles[1]}))]
                indcs += [i]
        return expectations, indcs

    def compute_log_pol2(self, f_s, indcs, policy, expectations, betas):
        gather = np.zeros_like(policy)
        for i in range(len(indcs)):
            gather[indcs[i]] += policy[i] * betas[i]
        log_pol2 = 0.
        for i in range(len(gather)):
            if gather[i]:
                log_pol2 += sym.Float(gather[i] * np.sign(expectations[i]), 10) * f_s[i]
        return log_pol2

    def compute_log_pol_diff(self, f_s, indcs, policy, expectations, betas, idx, sign, angles):
        x = self.x
        y = self.y
        gather = np.zeros_like(policy)
        for i in range(len(indcs)):
            gather[indcs[i]] += policy[i] * betas[i]
        diff_log_pol_x = sign * betas[idx] * float(f_s[idx].diff(x).subs({x: angles[0], y: angles[1]}))
        diff_log_pol_y = sign * betas[idx] * float(f_s[idx].diff(y).subs({x: angles[0], y: angles[1]}))
        for i in range(len(gather)):
            if gather[i]:
                diff_log_pol_x -= gather[i] * np.sign(expectations[i]) * float(
                    f_s[i].diff(x).subs({x: angles[0], y: angles[1]}))
                diff_log_pol_y -= gather[i] * np.sign(expectations[i]) * float(
                    f_s[i].diff(y).subs({x: angles[0], y: angles[1]}))
        return [diff_log_pol_x, diff_log_pol_y]

    def get_idx_beta(self, edge):
        i, j = edge
        if j < i:
            i, j = j, i
        return int(i * self.n - i * (i + 1) / 2 + j - i - 1)

    def get_binary(self, x, n):
        return 2 * np.array([int(b) for b in bin(x)[2:].zfill(n)], dtype=np.int32) - 1

    def bruteforce(self, J, n):
        maxi = -n
        idx = []
        for i in range(2 ** n - 1):
            z = self.get_binary(i, n)
            val = J.dot(z).dot(z)
            if val > maxi:
                maxi = val
                idx = [i]
            elif val == maxi:
                idx += [i]
        return maxi, idx

    def compute_f(self, J, i, j):
        x = self.x
        y = self.y
        prod1, prod2, prod3, prod4 = 1., 1., 1., 1.
        for k in range(len(J)):
            if k not in [i, j]:
                if J[i, k] - J[j, k]:
                    prod1 *= sym.cos(2 * x * (J[i, k] - J[j, k]))
                if J[i, k] + J[j, k]:
                    prod2 *= sym.cos(2 * x * (J[i, k] + J[j, k]))
                if J[i, k]:
                    prod3 *= sym.cos(2 * x * J[i, k])
                if J[j, k]:
                    prod4 *= sym.cos(2 * x * J[j, k])
        term = sym.Rational(0.5) * (sym.sin(2 * y) ** 2) * (prod1 - prod2) + sym.cos(2 * y) * sym.sin(2 * y) * sym.sin(
            2 * x * J[i, j]) * (prod3 + prod4)
        return term

    def generate_fs_h_actions(self, J, nodelist):
        f_s = []
        h = 0.
        action_space = []

        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    action_space += [(nodelist[i], nodelist[j])]
                    term = self.compute_f(J, i, j)
                    f_s += [term]
                    h += J[i, j] * term
        return f_s, h, action_space

    def expand_result(self, z_c, assignments, signs, nodelist):
        z_s = [self.get_binary(z, self.nc) for z in z_c]
        z_s = [np.array([z[nodelist.tolist().index(i)] if i in nodelist else 0 for i in range(self.n)], dtype=np.int32)
               for z in z_s]
        self.graph.reset()
        J = self.graph.get_G_numpy()
        ep_energies = []
        for i, assgn in enumerate(assignments[::-1]):
            for j, z in enumerate(z_s):
                z[assgn[1]] = signs[-i - 1] * z[assgn[0]]
                z_s[j] = z
            ep_energies.insert(0, J.dot(z_s[0]).dot(z_s[0]))
        ep_contribs = [ep_energies[i] - ep_energies[i + 1] for i in range(len(ep_energies) - 1)] + [ep_energies[-1]]
        return z_s, ep_energies, ep_contribs

    def softmax(self, values, beta):
        vals = np.array(values)
        vals -= np.amax(vals)
        vals = np.exp(beta * vals)
        return vals / np.sum(vals)

    def compute_extrema(self,
                        h,
                        search_space=[0, np.pi],
                        plot=False):
        x = self.x
        y = self.y

        # \gamma is x and \beta is y
        # E(x, y) = p * cos(4 * y) + q * sin(4 * x) + r, where p,q,r are complicated eqn's of x. Find p,q,r.
        r = (h.subs({x: x, y: np.pi/8}) + h.subs({x: x, y: -np.pi/8}))/2
        q = (h.subs({x: x, y: np.pi/8}) - h.subs({x: x, y: -np.pi/8}))/2
        p = h.subs({x: x, y: 0}) - r

        max_y_fun = r + sym.sqrt(p ** 2 + q ** 2)  # maximum of E(x,y) over all y's.
        fun = sym.lambdify([(x)], -max_y_fun, 'numpy')
        if plot:
            x_plot = scipy.linspace(0, 100*np.pi, 1000)
            plt.plot(x_plot, fun(x_plot))
            plt.show()
            plt.clf()
        ts = time.time()

        param_ranges = (slice(search_space[0], search_space[1], abs(search_space[1] - search_space[0]) / self.grid_N),)
        res_brute = optimize.brute(fun, param_ranges, full_output=True, finish=optimize.cobyla)

        solution = [res_brute[0]]

        q_val = q.subs({x: solution[0]})
        p_val = p.subs({x: solution[0]})
        y_val = 1 / 4 * (sym.atan2(q_val, p_val))
        assert (p_val * sym.cos(4 * y_val) >= 0)
        assert (q_val * sym.sin(4 * y_val) >= 0)
        solution = np.append(solution, y_val)

        extrema = res_brute[1]

        return solution, extrema

    def store_agent(self, pickle_path=None):

        if pickle_path is None:
            pickle_path = self.pickle_path
        pickle.dump(self, safe_open_w(pickle_path))


if __name__ == "__main__":
    sys_args = sys.argv
    if len(sys_args) > 1:
        if str(sys_args[3]) == 'None':
            nc = 8
            G = nx.random_regular_graph(d=8, n=14, seed=6121619833208740511)
            n = G.number_of_nodes()
            d = 8
            G = random_plus_minus_1_weights(graph=G, rs=np.random.RandomState(42), distribution='bimodal')
            G_name = '8d_14n_bimodaldist_6121619833208740511seed_17'

        else:
            G_name = str(sys_args[3])
            char1 = 'dist_'
            char2 = 'seed'
            char3 = 'd_'
            char4 = 'n'
            d = int(G_name[:G_name.find(char3)])
            n = int(G_name[G_name.find(char3)+2:G_name.find(char4)])
            generator_seed = int(G_name[G_name.find(char1)+5:G_name.find(char2)])
            distribution = G_name[G_name.find(char4)+2:G_name.find(char1)]
            G = nx.random_regular_graph(d=d, n=n, seed=generator_seed)
            G = random_weights(graph=G, rs=np.random.RandomState(42), type=distribution)
            nc = 8
            print(f'Solving {G_name}')

            
        if str(sys_args[10]) == 'None':
            init_angles1 = None
        else:
            init_angles1 = float(sys_args[10])
        init_angles2 = str(sys_args[11])
        if init_angles2 == 'None':
            init_angles2 = None

        agent = RQAOA_agent(n=n, nc=nc, G=G, d=d,
                            learning_rates=[float(sys_args[4]), float(sys_args[5])], gamma=float(sys_args[6]),
                            init_beta=[str(sys_args[7]), float(sys_args[8])],
                            init_angles=[str(sys_args[9]), init_angles1, init_angles2], batch_size=int(sys_args[12]),
                            normalize=(str(sys_args[13]) == 'True'), reward_fct=str(sys_args[14]),
                            run=str(sys_args[15]), idx=str(sys_args[1]), G_name=G_name)

        agent.play_train(nb_batches=int(sys_args[2]), train=(str(sys_args[16]) == 'True'))
