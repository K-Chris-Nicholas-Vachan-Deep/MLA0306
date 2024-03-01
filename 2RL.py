import numpy as np

news_articles = {
    0: 0.2,  # Efficiency of news article 0
    1: 0.4,  # Efficiency of news article 1
    2: 0.6,  # Efficiency of news article 2
    3: 0.8   # Efficiency of news article 3
}

V = {article: 0 for article in news_articles}

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
num_episodes = 1000  # Number of episodes

def choose_random_article():
    return np.random.choice(list(news_articles.keys()))

def choose_best_article():
    best_article = max(V, key=V.get)
    return best_article

for _ in range(num_episodes):
    current_article = choose_random_article()
    
    efficiency = news_articles[current_article]
    
    while True:
        next_article = choose_best_article()
        
        td_target = efficiency + gamma * V[next_article] - V[current_article]
        
        V[current_article] += alpha * td_target
        
        current_article = next_article
        
        if current_article == choose_best_article():
            break

best_article = choose_best_article()
print("Recommended News Article:", best_article)
print("Efficiency (Reward) of Recommended Article:", news_articles[best_article])
