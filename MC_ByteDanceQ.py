'''10 small balls are randomly divided into 12 boxes. You need to find the probability that exactly 10 boxes are empty with a program. 
   The program is required to simulate 100,000 times in order to calculate the probability with “brute-force”.'''
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

# Change seed each time
random.seed()

class byteDanceMC:
    def simple(self, iter_num):
        return self._run_naive_(iter_num)

    def medium_original(self, iter_num):
        return self._run_medium_(iter_num)

    # medium original
    def _run_medium_(self, iter_num):
        prob_part1, states_part1 = self.simulations_part1(10000)
        prob_part2 = self.simulations_part2(90000, states_part1)
        return prob_part1 * prob_part2

    def simulations_part1(self, number):
        five_empty_boxes = 0
        states = []
        for i in range(number):
            result = self._stimulation_(5, [0] * 12)
            if self._emptyBoxes_(result) >= 10:
                five_empty_boxes += 1
                states.append(result)
        return float(five_empty_boxes) / number, states

    def simulations_part2(self, number, states):
        ten_empty_boxes = 0
        for i in range(number):
            index = random.randint(0, len(states)-1)
            state = states[index][:]
            result = self._stimulation_(5, state)
            if self._emptyBoxes_(result) == 10:
                ten_empty_boxes += 1
        return float(ten_empty_boxes) / number

    def _stimulation_(self, balls=10, boxes=[0] * 12):
        for i in range(balls):
            index = random.randint(0, 11)
            boxes[index] += 1
        return boxes

    def _emptyBoxes_(self, boxes):
        number = 0
        for item in boxes:
            if item == 0:
                number += 1
        return number

    # Naive
    def _run_naive_(self, iter_num):
        '''Run naive approach.
        Note: this will probably result in zero as the true probability is so low.'''
        summ = 0
        for i in range(iter_num):
            summ += self._run_once_naive_()

        return summ/iter_num

    def _run_once_naive_(self):
        '''Return 0 for False, return 1 for True'''
        box_list = np.zeros(12)
        for i in range(10):
            index = random.randint(0, 11)
            box_list[index] += 1

        # Check if exactly 10 are empty
        count = 0
        for box in box_list:
            if box == 0:
                count += 1

        if count == 10:
            return 1
        else:
            return 0

    # TOTP
    def _run_totp_(self, iter_num):
        # Count number of times each event occur
        P_C = 0
        P_D = 0
        P_AC = 0
        P_AD = 0
        for i in range(iter_num):
            fir, sec, thr, four = self._run_once_totp_()
            P_C += fir
            P_D += sec
            P_AC += thr
            P_AD += four

        # Normalise to get probabilities
        P_C /= iter_num
        P_D /= iter_num
        P_AC /= iter_num
        P_AD /= iter_num
        # Return P(A) = P(A|C)P(C)+P(A|D)P(D)
        return P_AC*P_C + P_AD*P_D

    def _run_once_totp_(self):
        # Simulate 12 empty boxes
        box_list = np.zeros(12)

        # Throw 5 balls
        for i in range(5):
            index = random.randint(0, 11)
            box_list[index] += 1

        # Simulate P(A|C) - Equivalent to saying exactly two boxes are never empty
        box_list_AC = box_list.copy()
        box_list_AC[0] += 1
        box_list_AC[1] += 1
        # Simulate P(A|D) - Equivalent to saying exactly one box is never empty
        box_list_AD = box_list.copy()
        box_list_AD[0] += 1

        # Count empty boxes for each experiment
        count_0 = 0
        count_AC = 0
        count_AD = 0
        for box_0, box_AC, box_AD in zip(box_list, box_list_AC, box_list_AD):
            if box_0 == 0:
                count_0 += 1
            if box_AC == 0:
                count_AC += 1
            if box_AD == 0:
                count_AD += 1

        # Return tuple showing which events occured in this iteration
        result = [0, 0, 0, 0]
        if count_0 == 10:
            result[0] = 1
        if count_0 == 11:
            result[1] = 1
        if count_AC == 10:
            result[2] = 1
        if count_AD == 10:
            result[3] = 1
        return tuple(result)


class metaMCComparison:
    def __init__(self, mc_runs):
        self.byteDance = byteDanceMC()
        self.MC_runs = mc_runs

        self.med_mean = None
        self.totp_mean = None
        self.med_var = None
        self.totp_var = None

        self.mse_med = None
        self.mse_totp = None

        self.true_ans = 1.089387e-6

    def run_MC(self):
        runs = 100000

        medium_list = np.zeros(self.MC_runs)
        TOTP_list = np.zeros(self.MC_runs)

        for i in range(self.MC_runs):
            print(f"iteration: {i} of {self.MC_runs}")
            medium_list[i] = self.byteDance.medium_original(runs)
            TOTP_list[i] = self.byteDance._run_totp_(runs)

        np.save('medium_list', medium_list)
        np.save('TOTP_list', TOTP_list)

        self.med_mean = np.mean(medium_list)
        self.totp_mean = np.mean(TOTP_list)
        self.med_var = np.var(medium_list)
        self.totp_var = np.var(TOTP_list)

        self.mse_med = abs(medium_list-self.true_ans)
        self.mse_med = np.square(self.mse_med)
        self.mse_med = np.mean(self.mse_med)

        self.mse_totp = abs(TOTP_list-self.true_ans)
        self.mse_totp = np.square(self.mse_totp)
        self.mse_totp = np.mean(self.mse_totp)

        print(
            f"mse_med: {self.mse_med}, mse_totp: {self.mse_totp}")

        return self.med_mean, self.totp_mean, self.med_var, self.totp_var

    def plot(self):
        if not self.med_mean:
            print("run sims first pls")
            return

        mu = self.med_mean
        variance = self.med_var
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))

        mu = self.totp_mean
        variance = self.totp_var
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))

        plt.legend(['Medium', 'TOTP'])
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.yticks([])
        plt.xlabel('Probability of event A')
        plt.title('Medium vs TOTP-alternative MC PDF')
        plt.show()

    def plot_bar(self):
        TOTP_list = np.load('TOTP_list.npy')
        medium_list = np.load('medium_list.npy')

        mse_totp = abs(TOTP_list-self.true_ans)
        mse_totp = np.square(mse_totp)
        mse_totp = np.mean(mse_totp)

        mse_med = abs(medium_list-self.true_ans)
        mse_med = np.square(mse_med)
        mse_med = np.mean(mse_med)

        fig, ax = plt.subplots()
        labels = ('H-Sapien', 'TOTP')
        y_pos = np.arange(len(labels))
        performance = [mse_med, mse_totp]

        ax.barh(y_pos, performance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Mean Squared Error')
        ax.set_title('MSE of TOTP and H-Sapien Implementations')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.show()

    def plot_fromMem(self):
        TOTP_list = np.load('TOTP_list.npy')
        medium_list = np.load('medium_list.npy')
        plt.rcParams.update({'font.size': 18})
        plt.rc('axes', labelsize=14)    # fontsize of the x and y labels

        med_mean = np.mean(medium_list)
        totp_mean = np.mean(TOTP_list)
        eff_mean = np.mean(eff_list)
        med_var = np.var(medium_list)
        totp_var = np.var(TOTP_list)
        eff_var = np.var(eff_list)

        mu = med_mean
        variance = med_var
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))

        mu = eff_mean
        variance = eff_var
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))

        
        plt.rc('axes', titlesize=22)     # fontsize of the axes title
        
        plt.rc('xtick', labelsize=22) 
        plt.legend(['H-Sapien', 'TOTP'])
        
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.yticks([])
        plt.xlabel('Monte Carlo Prediction of P(A)')
        plt.show()


def simulation(iter_num):
    meta = metaMCComparison(iter_num)
    med_mean, totp_mean, med_var, totp_var = meta.run_MC()
    print(
        f'med_mean: {med_mean},totp_mean: {totp_mean},med_var: {med_var},totp_var: {totp_var}')
    meta.plot()


def plotMem():
    meta = metaMCComparison(None)
    meta.plot_fromMem()


if __name__ == '__main__':
    simulation(3000)
    
