import numpy as np
import matplotlib.pyplot as pyplot


def gamble(prob=0.4, state_num=99, threshold=1e-10):
    state_num = state_num
    policy = np.zeros((state_num, 1))
    value = np.zeros((state_num, 1))
    prob = prob
    counter = 0
    error = 100

    while error >= threshold:
        value_copy = value.copy()

        # s = {1, 2, ..., 99}
        for state in range(1, state_num+1):
            actions = range(0, np.minimum(state, state_num+1-state)+1)
            q_values = np.zeros((len(actions), 1))

            # a = {0, 1, ..., min(s, 100-s)}
            for a in actions:
                if (state + a) == 100:
                    head = prob
                else:
                    head = prob*value[state+a-1]

                if (state - a) == 0:
                    tail = 0
                else:
                    tail = (1-prob)*value[state-a-1]

                q_values[a] = head + tail

            value[state-1] = np.amax(q_values)
            idxs = np.where(q_values == value[state-1])
            policy[state-1] = int(np.random.choice(idxs[0], 1)[0])

        counter += 1
        error = np.sum((np.array(value) - np.array(value_copy))**2)

    return policy, value


def main():
    fig = plt.figure(figsize=(12, 8))
    axes = []

    state_num = 99
    prob = 0.25

    # Ph = 0.25, plot value estimation
    axes.append(fig.add_subplot(2, 2, 1))
    for i in range(500):
        policy, value = gamble(prob=prob, state_num=state_num)
        axes[-1].plot(range(1, state_num+1), policy[:, 0], 'o', color='blue')
        axes[-1].set_xlabel('Capital', fontsize=16)
        axes[-1].set_ylabel('Final Policy (Stake)', fontsize=16)
        axes[-1].set_title('Ph = {}'.format(prob), fontsize=16)

    # Ph = 0.25, plot policy
    axes.append(fig.add_subplot(2, 2, 2))
    axes[-1].plot(value)
    axes[-1].set_xlabel('Capital', fontsize=16)
    axes[-1].set_ylabel('Value Estimate', fontsize=16)
    axes[-1].set_title('Ph = {}'.format(prob), fontsize=16)

    prob = 0.55

    # Ph = 0.55, plot value estimation
    axes.append(fig.add_subplot(2, 2, 3))
    for i in range(100):
        policy, value = gamble(prob=prob, state_num=state_num)
        axes[-1].plot(range(1, state_num+1), policy[:, 0], 'o', color='blue')
        axes[-1].set_xlabel('Capital', fontsize=16)
        axes[-1].set_ylabel('Final Policy (Stake)', fontsize=16)
        axes[-1].set_title('Ph = {}'.format(prob), fontsize=16)

    # Ph = 0.55, plot policy
    axes.append(fig.add_subplot(2, 2, 4))
    axes[-1].plot(value)
    axes[-1].set_xlabel('Capital', fontsize=16)
    axes[-1].set_ylabel('Value Estimate', fontsize=16)
    axes[-1].set_title('Ph = {}'.format(prob), fontsize=16)

    fig.tight_layout()
    plt.show()
    fig.savefig('gamble.png', bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    main()
