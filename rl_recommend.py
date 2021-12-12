import pandas as pd
import os
import numpy as np
import itertools
import scipy.spatial.distance as distlib
import random as rand
import sys
import time

class RecommendationSystem:
    def __init__(self, dataset_with_bins, initial_songs, n_features, n_bins):
        '''
        We assume here that the columns of the dataset here are already the binary percentile bins for all features 
        For sanity check: n_features x n_bins = length(dataset_with_bins)
        '''
        self.data = dataset_with_bins # assuming this is the whole dataset split that we want to work with
        self.data.index = np.arange(np.shape(self.data)[0])
        self.initial_songs = initial_songs # assuming this is still a pandas df of k_s rows, but only containing the song rows that the user prefers
        self.n_features = n_features
        self.n_bins = n_bins
        self.k_s = np.shape(self.initial_songs)[0]
        self.k_t = 10 # for now, just queue them 10 songs to choose from
        
        # Initialize preferences
        self.init_song_preferences()
        self.init_transition_preferences()

    def init_song_preferences(self):
        # Initialize preference array
        self.phi_s = (1/(self.k_s + 1)) * np.ones((self.n_features * self.n_bins, 1))
        tmp = (np.sum(self.initial_songs.values > 0, axis = 0)) * (1/(self.k_s + 1))
        self.phi_s = self.phi_s + np.reshape(tmp, (len(tmp), 1))

    def theta_t(self, idx_a, idx_b):
        '''
        Input: indices of songs a and b within the provided dataset (int)

        Output: vector theta_t, assuming the feature sequence of 1-i, 1-2, ..., 1-n_bins, 2-1, ..., n_bins-1, n_bins-2, ..., n_bins-n_bins
        '''
        indices = np.array([], dtype=int)
        for i in range(self.n_features):
            a_bin_idx = np.where(self.data.loc[idx_a][self.data.columns[i*self.n_bins:(i+1)*self.n_bins]] == 1.0)[0]
            b_bin_idx = np.where(self.data.loc[idx_b][self.data.columns[i*self.n_bins:(i+1)*self.n_bins]] == 1.0)[0]
            indices = np.append(indices, int(i*a_bin_idx + b_bin_idx))
        out = np.zeros((self.n_bins * self.n_bins * self.n_features, 1))
        out[indices] = 1
        return out

    def init_transition_preferences(self):
        # Initialize user preference vector
        self.phi_t = (1/(self.k_t + 1)) * np.ones((self.n_features * self.n_bins * self.n_bins, 1))

        # Take the upper-median preference split
        self.Rs = np.sum(np.matmul(self.data.values, self.phi_s), axis=1)
        self.Mstar = self.data
        self.Mstar['Rs'] = self.Rs
        self.Mstar.sort_values('Rs', inplace = True, ascending = False)
        self.Mstar = self.Mstar[:np.shape(self.Mstar)[0] // 2]

        # Generate 10th percentile distance of all pairwise distances from M (not M*)
        diff = distlib.pdist(self.data.values, 'cosine') #taking cosine distance metric between songs, not 100% sure on this
        delta = np.percentile(diff, 10, axis=0)

        # Generate a representative subset of M*
        representatives = self.delta_medoids(self.Mstar, delta)
        song_prev = np.random.choice(representatives)
        for i in range(self.k_t):
            song = np.random.choice(representatives) # for now, assume that the user picks randomly from the preferred dataset. This is obviously a wrong representation of transition preferences
            self.phi_t += 1/(self.k_t + 1) * self.theta_t(song_prev, song)
            song_prev = song
        pass
        
    def one_shot_delta(self, data, delta, clusters):
        # Remember to change the distance metric in case we change delta distance definition
        for index, row in data.iterrows():
            dist = 1e6
            representatives = clusters.keys()
            for rep in representatives:
                if distlib.cosine(row, rep) <= dist:
                    representative = int(rep)
                    dist = distlib.cosine(row, data.loc[rep])
            if dist <= delta:
                clusters[representative] = np.append(clusters[representative], index)
            else: 
                representative = index
                clusters[representative] = np.array([representative])

        out = clusters.keys()
        return clusters

    def delta_medoids(data, delta):
        distances = distlib.squareform(distances)
        np.fill_diagonal(distances, 0)
        exit_loop = False
        i = 0
        clusters = {}
        while not exit_loop:
            i +=1
            clusters = one_shot_delta(data, delta, clusters)
            if 1 != i:
                representatives_prev = representatives
            else:
                representatives_prev = np.array([])
            representatives = np.array([])
            for cluster in clusters.items():
                cluster = cluster[1]
                cluster_dists = distlib.pdist(data.loc[cluster], 'cosine')
                cluster_dists = distlib.squareform(cluster_dists)
                argmin = np.argmin(np.sum(cluster_dists, axis=0))
                representatives = np.append(representatives, cluster[argmin])
            if np.array_equal(np.sort(representatives), np.sort(representatives_prev)):
                exit_loop = True
        return representatives

class MDP:
    def __init__(self):

        df = pd.read_csv("MSD.csv", index_col=None)

        self.features = ['duration', 'key_confidence', 'end_of_fade_in', 'mode_confidence', 'start_of_fade_out', 'tempo',
                    'artist_hotttnesss', 'song_hotttnesss']

        for feature in self.features:
          filter = df[feature] > 0
          df = df[filter]
        
        self.filtered_df = df[self.features]
        self.filtered_df.dropna(inplace=True)
        self.filtered_df = self.filtered_df.reset_index(drop=True)
        self.n_rows = len(self.filtered_df.index)
        self.n_songs = self.n_rows
        self.n_bins = 10
        self.n_features = len(self.features)
        self.id_song_to_vec = dict()
        self.songs = set()        
        self.song_id_to_song_row = []
        self.song_priors = []

        print(self.n_songs)
        song_id = 0
      
        for song, row in self.filtered_df.iterrows():
            self.id_song_to_vec[song_id] = np.zeros(80)
            self.songs.add(song_id)
            self.song_id_to_song_row.append(row)
            self.song_priors.append(1)
            song_id += 1
            # row_id += 1
            print(song)


        self.songs_list = [i for i in range(self.n_songs)]

        print(self.features)
        for f in range(self.n_features):

            feature = self.features[f]
            print(feature)
            sorted_df = self.filtered_df.sort_values(by=[feature])
            i = 0
            for idx, row in sorted_df.iterrows():
                bin = i * self.n_bins // self.n_rows
                self.id_song_to_vec[idx][f * 10 + bin] = 1
                i += 1

        print("ssss")

        df_input = pd.DataFrame.from_dict(self.id_song_to_vec, orient='index')
        # Select songs that the user prefers (randomly selecting 10 for now, change later)
        self.init_prefs = df_input.loc[rand.sample(self.songs, 5)]
        print("fsdf")

        time_start = time.time()
        rs = RecommendationSystem(df_input, self.init_prefs, self.n_features, self.n_bins)
        

        print("fdsd")
        self.phi_s = rs.phi_s.reshape(80)
        self.phi_t = rs.phi_t.reshape(800)
        print("fffff")
        # self.transition_reward = np.zeros((self.n_songs, self.n_songs))
        # for i in range(self.n_songs):
        #   for j in range(self.n_songs):
        #     self.transition_reward[i,j] = np.dot(self.phi_t, self.get_theta_t(i, j))


    def get_next_state(self, state, action):
      return state + [action]
      
    def get_valid_actions(self, state):
        # All actions are invalid by default
        valid_actions = [0] * self.n_songs

        for song in range(self.n_songs):
            if song not in state:
                valid_actions[song] = 1

        return valid_actions

    def state_to_input(self, state):
      input = np.ones(800)*-1
      for i in range(len(state)):
        song = state[i]
        input[i*80:(i+1)*80] = self.id_song_to_vec[song]
      return input

    def get_reward(self, final_state):
      if len(final_state) < 10:
        return None
      elif len(final_state) == 10:
        state = []
        trajectory_states = [state]
        trajectory_actions = []
        for song in final_state:
          state = self.get_next_state(state, song)
          trajectory_states.append(state)
          trajectory_actions.append(song)
          
        return self.payoff_trajectory(trajectory_states, trajectory_actions, 11)
      else:
        print("Error: length > 10")
        return None


    def get_theta_t(self, s1, s2):
        theta_t = np.zeros(800)
        for f in range(len(self.features)):
            for i in range(10):
                for j in range(10):
                    if self.id_song_to_vec[s1][f*10 + i] == 1 and  self.id_song_to_vec[s2][f*10 + j]:
                        theta_t[f*100 + i*10 + j] = 1
        return  theta_t

    def theta_s(self, s):
        return self.id_song_to_vec[s]

    def R(self, s, a):
        Rs = np.dot(self.phi_s, self.theta_s(a))
        Rt = 0
        for i in range(len(s)):
            song_past = s[i]
            Rt += 1/((i+1)*(i+1))* np.dot(self.phi_t, self.get_theta_t(song_past, a))
        return Rs + Rt


    def payoff_trajectory(self, trajectory_states, trajectory_actions, real_action_len):
      payoff = 0

      for t in range(len(trajectory_actions)):
        payoff += (1)**(max(t-real_action_len,0)) * self.R(trajectory_states[t], trajectory_actions[t])
      return payoff

    def MC_value(self, s):
      count = 0
      sum_values = 0
      state = s
      # Past episodes
      episode_states = [[]]
      episode_actions = []

      for i in range(10):
        if i < len(s):
          state = s[:i+1]
          action = s[i]
        else:
          action_probs = list(model.song_priors)
          for song_id in range(model.n_songs):
            if song_id in state:
              action_probs[song_id] = 0

          action_probs = np.array(action_probs)/np.sum(np.array(action_probs))
          action = np.random.choice(model.songs_list, 1, p=action_probs)[0]
          state = episode_states[-1] + [action]
        episode_states.append(state)
        episode_actions.append(action)

    #   print("MC estimate:", s, episode_states, episode_actions)
          
          
      # Set MC 
      payoff = self.payoff_trajectory(episode_states, episode_actions, len(s)-2)
      
      return payoff

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

  

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.state = state
        for song_id in range(len(action_probs)):
              self.children[song_id] = Node(action_probs[song_id])

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())



model = MDP()

# class MDP:
#     def __init__(self):

#         df = pd.read_csv("MSD.csv", index_col=None)
#         # filter_duration = df['duration'] > 0
#         # filter_end_fade_in = df['end_of_fade_in'] > 0
#         # filter_key_confidence = df['key_confidence'] > 0
#         # # filter_loudness = df['key_confidence'] > 0
#         # filter_mode_confidence = df['mode_confidence'] > 0
#         # filter_start_fade_out = df['start_of_fade_out'] > 0
#         # filter_tempo = df['tempo'] > 0
#         # filter_artist_hottness = df['artist_hotttnesss'] > 0
#         filter_song_hottness = df['song_hotttnesss'] > 0
#         # self.filtered_df = df[filter_duration][filter_end_fade_in][filter_key_confidence][filter_mode_confidence][filter_start_fade_out][
#             # filter_tempo][filter_artist_hottness][filter_song_hottness]
#         self.features = ['duration', 'key_confidence', 'end_of_fade_in', 'mode_confidence', 'start_of_fade_out', 'tempo',
#                     'artist_hotttnesss', 'song_hotttnesss']

#         self.filtered_df = df[self.features]
#         self.filtered_df.dropna(inplace=True)
#         self.n_rows = len(self.filtered_df.index)
#         self.n_bins = 10
#         self.n_features = len(self.features)
#         self.id_song_to_vec = dict()
#         self.songs = set()
#         self.song_priors = {}

#         for song, row in self.filtered_df.iterrows():
#             self.id_song_to_vec[song] = np.zeros(80)
#             self.songs.add(song)
#             self.song_priors[song] = 1

#         for f in range(self.n_features):
#             feature = self.features[f]
#             sorted_df = self.filtered_df.sort_values(by=[feature])
#             i = 0
#             for index, row in sorted_df.iterrows():
#                 bin = i * self.n_bins // self.n_rows
#                 i += 1
#                 self.id_song_to_vec[index][f * 10 + bin] = 1

#         df_input = pd.DataFrame.from_dict(self.id_song_to_vec, orient='index')
#         # Select songs that the user prefers (randomly selecting 10 for now, change later)
#         self.init_prefs = df_input.loc[rand.sample(self.songs, 10)]

#         # TODO: The dataset still needs to be cleaned from all bs, all NaNs, all string entries         
#         rs = RecommendationSystem(df_input, self.init_prefs, self.n_features, self.n_bins)

#         self.phi_s = rs.phi_s
#         self.phi_t = rs.phi_t

#     def theta_t(self, s1, s2):

#         theta_t = np.zeros(800)
#         for f in range(len(self.features)):
#             for i in range(10):
#                 for j in range(10):
#                     if self.id_song_to_vec[s1][f*10 + i] == 1 and  self.id_song_to_vec[s2][f*10 + j]:
#                         theta_t[f*100 + i*10 + j] = 1
#         return  theta_t

#     def theta_s(self, s):
#         return self.id_song_to_vec[s]

#     def R(self, s, a):
#         Rs = np.dot(self.phi_s, self.theta_s(a))
#         Rt = 0
#         for i in range(len(s)):
#             song_past = s[i]
#             Rt += 1/((i+1)*(i+1))* np.dot(self.phi_t, self.theta_t(song_past, a))
#         return Rs + Rt


# class Node:
#     def __init__(self, prior):
#         self.visit_count = 0
#         self.prior = prior
#         self.value_sum = 0
#         self.children = {}
#         self.state = None

#     def expanded(self):
#         return len(self.children) > 0

#     def value(self):
#         if self.visit_count == 0:
#             return 0
#         return self.value_sum / self.visit_count

#     def select_action(self, temperature):
#         """
#         Select action according to the visit count distribution and the temperature.
#         """
#         visit_counts = np.array([child.visit_count for child in self.children.values()])
#         actions = [action for action in self.children.keys()]
#         if temperature == 0:
#             action = actions[np.argmax(visit_counts)]
#         elif temperature == float("inf"):
#             action = np.random.choice(actions)
#         else:
#             # See paper appendix Data Generation
#             visit_count_distribution = visit_counts ** (1 / temperature)
#             visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
#             action = np.random.choice(actions, p=visit_count_distribution)

#         return action

#     def select_child(self):
#         """
#         Select the child with the highest UCB score.
#         """
#         best_score = -np.inf
#         best_action = -1
#         best_child = None

#         for action, child in self.children.items():
#             score = ucb_score(self, child)
#             if score > best_score:
#                 best_score = score
#                 best_action = action
#                 best_child = child

#         return best_action, best_child

#     def expand(self, state, action_probs):
#         """
#         We expand a node and keep track of the prior policy probability given by neural network
#         """
#         self.state = state
#         for song, prob in action_probs.items():
#             if prob != 0:
#                 self.children[song] = Node(prob)

#     def __repr__(self):
#         """
#         Debugger pretty print node info
#         """
#         prior = "{0:.2f}".format(self.prior)
#         return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())