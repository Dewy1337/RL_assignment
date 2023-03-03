library(contextual)
library(ggnormalviolin)
library(FactoMineR)
library(haven)
library(reshape2)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr,
               tidyr,
               ggplot2,
               reshape2,
               latex2exp,
               devtools,
               BiocManager)

# set the seed
set.seed(1234)

get_results <- function(res){
  return(res$data %>%
           select(t, sim, choice, reward, agent))
}


get_maxObsSims <- function(result){
  return(result%>%
           group_by(sim, agent) %>%
           summarise(max_t = max(t)))
}


show_results <- function(df_results, max_obs){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  df_history_agg <- df_results %>%
    group_by(sim)%>% # group by simulation
    mutate(cumulative_reward = cumsum(reward))%>% # calculate, per sim, cumulative reward over time
    group_by(t) %>% # group by timestep 
    summarise(avg_cumulative_reward = mean(cumulative_reward), # average cumulative reward
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>% # SE + Confidence interval
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <= max_obs)
  
  
  # define the legend of the plot
  legend <- c("Avg." = "orange", "95% CI" = "gray") # set legend
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward))+ 
    geom_line(size=1.5,aes(color="Avg."))+ # add line 
    geom_ribbon(aes(ymin=ifelse(cumulative_reward_lower_CI<0, 0,cumulative_reward_lower_CI),
                    # add confidence interval
                    ymax=cumulative_reward_upper_CI,
                    color = "95% CI"
    ), # 
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color='Metric')+ # add titles
    scale_color_manual(values=legend)+ # add legend
    theme_bw()+ # set the theme
    theme(text = element_text(size=16)) # enlarge text
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2, df_history_agg = df_history_agg))
}


show_results_multipleagents <- function(df_results, max_obs){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  
  # data.frame aggregated for two versions: 20 and 40 arms
  df_history_agg <- df_results %>%
    group_by(agent, sim)%>% # group by number of arms, the sim
    mutate(cumulative_reward = cumsum(reward))%>% # calculate cumulative sum
    group_by(agent, t) %>% # group by number of arms, the t
    summarise(avg_cumulative_reward = mean(cumulative_reward),# calc cumulative reward, se, CI
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>%
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <=max_obs)
  
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward, color =agent))+
    geom_line(size=1.5)+
    geom_ribbon(aes(ymin=cumulative_reward_lower_CI , 
                    ymax=cumulative_reward_upper_CI,
                    fill = agent,
    ),
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color ='c', fill='c')+
    theme_bw()+
    theme(text = element_text(size=16))
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}


classicPCA <- function(mX){
  #perform pca
  pca <- prcomp(mX)
  pca_rotation <- pca$rotation
  
  eigenvalues <- pca$sdev^2
  print(sum(eigenvalues[1:9])/sum(eigenvalues))
  
  #print largest loadings
  print(sort(abs(pca_rotation[,1])))
  print(sort(abs(pca_rotation[,2])))
  print(sort(abs(pca_rotation[,3])))
  
  #create scree plot
  print(fviz_eig(pca))
  return(pca)
}

melt_dataframe <- function(df){
  df <- melt(as.matrix(df))
  df <- df[order(df$Var1),]
  return(df)
}

run_simulator <- function(agents){
  # Create Similator
  simulator          <- Simulator$new(agents, # set our agents
                                      horizon= size_sim, # set the sizeof each simulation
                                      do_parallel = TRUE, # run in parallel for speed
                                      simulations = n_sim, # simulate it n_sim times,
                                      
  )
  
  # run the simulator object
  history_stocks <- simulator$run()
  res <- get_results(history_stocks)
  max_obs <- min(get_maxObsSims(res)$max_t)
  show_results_multipleagents(res, max_obs)
}


#Comparison UCB lin UCB
df_returns <- read.csv("returns.csv", row.names = 1)[15:6023,]
df_rewards <- read.csv("reward.csv", row.names = 1)[15:6023,]
df_OBV <- read.csv("OBV.csv", row.names = 1)[15:6023,]
df_RSI <- read.csv("RSI.csv", row.names = 1)
df_implied_vol <- read.csv("implied_vol_C_7_30.csv", row.names = 1)[16:6024,]

iT <- dim(df_rewards)[1]
iN <- dim(df_rewards)[2]

df_rewards <- melt_dataframe(df_rewards)
df_returns <- melt_dataframe(df_returns)
df_OBV <- melt_dataframe(df_OBV)
df_RSI <- melt_dataframe(df_RSI)
df_implied_vol <- melt_dataframe(df_implied_vol)

df <- data.frame(df_rewards, df_returns$value, df_OBV$value, df_RSI$value, df_implied_vol$value)
row.names(df) <- NULL
colnames(df) <- c("timestamp", "symbol", "reward", "return", "OBV", "RSI", "implied_vol")
df$symbol <- as.integer(df$symbol)

# Create the bandit
bandit <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df, randomize = FALSE, replacement = FALSE)
bandit_context <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + implied_vol + RSI + OBV, data = df, randomize = FALSE, replacement = FALSE)

# Define the UCB policy
alpha = 0.1
UCB_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_context_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_context_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_context_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_context_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_context_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)
UCB_context_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)

# Define the TS policy
nu <- 0.01
TS_04 <- ContextualLinTSPolicy$new(v=0.4)
TS_context_04 <- ContextualLinTSPolicy$new(v=0.4)
TS_03 <- ContextualLinTSPolicy$new(v=0.3)
TS_context_03 <- ContextualLinTSPolicy$new(v=0.3)
TS_02 <- ContextualLinTSPolicy$new(v=0.2)
TS_context_02 <- ContextualLinTSPolicy$new(v=0.2)
TS_01 <- ContextualLinTSPolicy$new(v=0.1)
TS_context_01 <- ContextualLinTSPolicy$new(v=0.1)
TS_005 <- ContextualLinTSPolicy$new(v=0.05)
TS_context_005 <- ContextualLinTSPolicy$new(v=0.05)
TS_001 <- ContextualLinTSPolicy$new(v=0.01)
TS_context_001 <- ContextualLinTSPolicy$new(v=0.01)

# Create the agent
agent_UCB_04 <- Agent$new(policy = UCB_04, bandit = bandit, name = "UCB a=0.4")
agent_context_UCB_04 <- Agent$new(policy = UCB_context_04, bandit = bandit_context, name = "CUCB a=0.4")
agent_UCB_03 <- Agent$new(policy = UCB_03, bandit = bandit, name = "UCB a=0.3")
agent_context_UCB_03 <- Agent$new(policy = UCB_context_03, bandit = bandit_context, name = "CUCB a=0.3")
agent_UCB_02 <- Agent$new(policy = UCB_02, bandit = bandit, name = "UCB a=0.2")
agent_context_UCB_02 <- Agent$new(policy = UCB_context_02, bandit = bandit_context, name = "CUCB a=0.2")
agent_UCB_01 <- Agent$new(policy = UCB_01, bandit = bandit, name = "UCB a=0.1")
agent_context_UCB_01 <- Agent$new(policy = UCB_context_01, bandit = bandit_context, name = "CUCB a=0.1")
agent_UCB_005 <- Agent$new(policy = UCB_005, bandit = bandit, name = "UCB a=0.05")
agent_context_UCB_005 <- Agent$new(policy = UCB_context_005, bandit = bandit_context, name = "CUCB a=0.05")
agent_UCB_001 <- Agent$new(policy = UCB_001, bandit = bandit, name = "UCB a=0.01")
agent_context_UCB_001 <- Agent$new(policy = UCB_context_001, bandit = bandit_context, name = "CUCB 0.01")


agent_TS_04 <- Agent$new(policy = TS_04, bandit = bandit, name = "TS v=0.4")
agent_context_TS_04 <- Agent$new(policy = TS_context_04, bandit = bandit_context, name = "CTS v=0.4")
agent_TS_03 <- Agent$new(policy = TS_03, bandit = bandit, name = "TS v=0.3")
agent_context_TS_03 <- Agent$new(policy = TS_context_03, bandit = bandit_context, name = "CTS v=0.3")
agent_TS_02 <- Agent$new(policy = TS_02, bandit = bandit, name = "TS v=0.2")
agent_context_TS_02 <- Agent$new(policy = TS_context_02, bandit = bandit_context, name = "CTS v=0.2")
agent_TS_01 <- Agent$new(policy = TS_01, bandit = bandit, name = "TS v=0.1")
agent_context_TS_01 <- Agent$new(policy = TS_context_01, bandit = bandit_context, name = "CTS v=0.1")
agent_TS_005 <- Agent$new(policy = TS_005, bandit = bandit, name = "TS v=0.05")
agent_context_TS_005 <- Agent$new(policy = TS_context_005, bandit = bandit_context, name = "CTS v=0.05")
agent_TS_001 <- Agent$new(policy = TS_001, bandit = bandit, name = "TS v=0.01")
agent_context_TS_001 <- Agent$new(policy = TS_context_001, bandit = bandit_context, name = "CTS v=0.01")

#Portfolios
df_portfolio_returns <- read.csv("portfolio_returns.csv", row.names = 1)[14:6022,]
df_portfolio_rewards <- read.csv("portfolio_rewards.csv", row.names = 1)[14:6022,]
df_portfolio_RSI <- read.csv("portfolio_rsi.csv", row.names = 1)
df_portfolio_implied_vol <- read.csv("portfolio_vol.csv", row.names = 1)[14:6022,]

iT <- dim(df_portfolio_rewards)[1]
iN <- dim(df_portfolio_rewards)[2]

df_portfolio_rewards <- melt_dataframe(df_portfolio_rewards)
df_portfolio_returns <- melt_dataframe(df_portfolio_returns)
df_portfolio_RSI <- melt_dataframe(df_portfolio_RSI)
df_portfolio_implied_vol <- melt_dataframe(df_portfolio_implied_vol)

df_portfolio <- data.frame(df_portfolio_rewards, df_portfolio_returns$value, df_portfolio_RSI$value, df_portfolio_implied_vol$value)
row.names(df_portfolio) <- NULL
colnames(df_portfolio) <- c("timestamp", "symbol", "reward", "return", "RSI", "implied_vol")
df_portfolio$symbol <- as.integer(df_portfolio$symbol)

# Create the bandit
bandit_portfolio <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio, randomize = FALSE, replacement = FALSE)
bandit_portfolio_context <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio, randomize = FALSE, replacement = FALSE)

# Define the UCB policy
UCB_portfolio_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_portfolio_context_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_portfolio_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_portfolio_context_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_portfolio_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_portfolio_context_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_portfolio_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_portfolio_context_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_portfolio_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_portfolio_context_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_portfolio_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)
UCB_portfolio_context_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)

#Thompson
TS_portfolio_04 <- ContextualLinTSPolicy$new(v=0.4)
TS_portfolio_context_04 <- ContextualLinTSPolicy$new(v=0.4)
TS_portfolio_03 <- ContextualLinTSPolicy$new(v=0.3)
TS_portfolio_context_03 <- ContextualLinTSPolicy$new(v=0.3)
TS_portfolio_02 <- ContextualLinTSPolicy$new(v=0.2)
TS_portfolio_context_02 <- ContextualLinTSPolicy$new(v=0.2)
TS_portfolio_01 <- ContextualLinTSPolicy$new(v=0.1)
TS_portfolio_context_01 <- ContextualLinTSPolicy$new(v=0.1)
TS_portfolio_005 <- ContextualLinTSPolicy$new(v=0.05)
TS_portfolio_context_005 <- ContextualLinTSPolicy$new(v=0.05)
TS_portfolio_001 <- ContextualLinTSPolicy$new(v=0.01)
TS_portfolio_context_001 <- ContextualLinTSPolicy$new(v=0.01)


# Create the agents
agent_portfolio_UCB_04 <- Agent$new(policy = UCB_portfolio_04, bandit = bandit_portfolio, name = "UCB portfolio a=0.4")
agent_portfolio_context_UCB_04 <- Agent$new(policy = UCB_portfolio_context_04, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.4")
agent_portfolio_UCB_03 <- Agent$new(policy = UCB_portfolio_03, bandit = bandit_portfolio, name = "UCB portfolio a=0.3")
agent_portfolio_context_UCB_03 <- Agent$new(policy = UCB_portfolio_context_03, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.3")
agent_portfolio_UCB_02 <- Agent$new(policy = UCB_portfolio_02, bandit = bandit_portfolio, name = "UCB portfolio a=0.2")
agent_portfolio_context_UCB_02 <- Agent$new(policy = UCB_portfolio_context_02, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.2")
agent_portfolio_UCB_01 <- Agent$new(policy = UCB_portfolio_01, bandit = bandit_portfolio, name = "UCB portfolio a=0.1")
agent_portfolio_context_UCB_01 <- Agent$new(policy = UCB_portfolio_context_01, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.1")
agent_portfolio_UCB_005 <- Agent$new(policy = UCB_portfolio_005, bandit = bandit_portfolio, name = "UCB portfolio a=0.05")
agent_portfolio_context_UCB_005 <- Agent$new(policy = UCB_portfolio_context_005, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.05")
agent_portfolio_UCB_001 <- Agent$new(policy = UCB_portfolio_001, bandit = bandit_portfolio, name = "UCB portfolio a=0.01")
agent_portfolio_context_UCB_001 <- Agent$new(policy = UCB_portfolio_context_001, bandit = bandit_portfolio_context, name = "CUCB portfolio a=0.01")

agent_portfolio_TS_04 <- Agent$new(policy = TS_portfolio_04, bandit = bandit_portfolio, name = "TS portfolio v=0.4")
agent_portfolio_context_TS_04 <- Agent$new(policy = TS_portfolio_context_04, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.4")
agent_portfolio_TS_03 <- Agent$new(policy = TS_portfolio_03, bandit = bandit_portfolio, name = "TS portfolio v=0.3")
agent_portfolio_context_TS_03 <- Agent$new(policy = TS_portfolio_context_03, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.3")
agent_portfolio_TS_02 <- Agent$new(policy = TS_portfolio_02, bandit = bandit_portfolio, name = "TS portfolio v=0.2")
agent_portfolio_context_TS_02 <- Agent$new(policy = TS_portfolio_context_02, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.2")
agent_portfolio_TS_01 <- Agent$new(policy = TS_portfolio_01, bandit = bandit_portfolio, name = "TS portfolio v=0.1")
agent_portfolio_context_TS_01 <- Agent$new(policy = TS_portfolio_context_01, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.1")
agent_portfolio_TS_005 <- Agent$new(policy = TS_portfolio_005, bandit = bandit_portfolio, name = "TS portfolio v=0.05")
agent_portfolio_context_TS_005 <- Agent$new(policy = TS_portfolio_context_005, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.05")
agent_portfolio_TS_001 <- Agent$new(policy = TS_portfolio_001, bandit = bandit_portfolio, name = "TS portfolio v=0.01")
agent_portfolio_context_TS_001 <- Agent$new(policy = TS_portfolio_context_001, bandit = bandit_portfolio_context, name = "CTS portfolio v=0.01")

# Simulator settings
size_sim=100000
n_sim=14

# run simulator UCB individual stocks
run_simulator(list(agent_UCB_04, agent_UCB_03,  agent_UCB_02, agent_UCB_01, agent_UCB_005, agent_UCB_001))

# run simulator UCB individual stocks context
run_simulator(list(agent_context_UCB_04, agent_context_UCB_03,  agent_context_UCB_02, agent_context_UCB_01, agent_context_UCB_005, agent_context_UCB_001))

# run simulator TS individual stocks
run_simulator(list(agent_TS_04, agent_TS_03,  agent_TS_02, agent_TS_01, agent_TS_005, agent_TS_001))

# run simulator TS individual stocks context
run_simulator(list(agent_context_TS_04, agent_context_TS_03,  agent_context_TS_02, agent_context_TS_01, agent_context_TS_005, agent_context_TS_001))

# run simulator UCB portfolio
run_simulator(list(agent_portfolio_UCB_04, agent_portfolio_UCB_03,  agent_portfolio_UCB_02, agent_portfolio_UCB_01, agent_portfolio_UCB_005, agent_portfolio_UCB_001))

# run simulator TS portfolio
run_simulator(list(agent_portfolio_TS_04, agent_portfolio_TS_03,  agent_portfolio_TS_02, agent_portfolio_TS_01, agent_portfolio_TS_005, agent_portfolio_TS_001))

# run simulator UCB portfolio context
run_simulator(list(agent_portfolio_context_UCB_04, agent_portfolio_context_UCB_03,  agent_portfolio_context_UCB_02, agent_portfolio_context_UCB_01, agent_portfolio_context_UCB_005, agent_portfolio_context_UCB_001))

# run simulator TS portfolio context
run_simulator(list(agent_portfolio_context_TS_04, agent_portfolio_context_TS_03,  agent_portfolio_context_TS_02, agent_portfolio_context_TS_01, agent_portfolio_context_TS_005, agent_portfolio_context_TS_001))

best_a = 0.3
best_a_context = 0.3
best_nu = 0.05
best_nu_context = 0.4

best_a_portfolio = 0.3
best_a_portfolio_context = 0.1
best_nu_portfolio = 0.3
best_nu_portfolio_context = 0.01

# run simulator best models
run_simulator(list(agent_UCB_03, agent_context_UCB_03, agent_TS_005, agent_context_TS_04,
                   agent_portfolio_UCB_03, agent_portfolio_context_UCB_01, 
                   agent_portfolio_TS_03, agent_portfolio_context_TS_001))

#Experiment for amount of portfolios
df_portfolio_5 <- df_portfolio[df_portfolio$symbol %in% sample(1:25, 5), ]
df_portfolio_5$symbol <- as.factor(df_portfolio_5$symbol)
df_portfolio_5$symbol <- as.integer(df_portfolio_5$symbol)
df_portfolio_10 <- df_portfolio[df_portfolio$symbol %in% sample(1:25, 10), ]
df_portfolio_10$symbol <- as.factor(df_portfolio_10$symbol)
df_portfolio_10$symbol <- as.integer(df_portfolio_10$symbol)
df_portfolio_15 <- df_portfolio[df_portfolio$symbol %in% sample(1:25, 15), ]
df_portfolio_15$symbol <- as.factor(df_portfolio_15$symbol)
df_portfolio_15$symbol <- as.integer(df_portfolio_15$symbol)
df_portfolio_20 <- df_portfolio[df_portfolio$symbol %in% sample(1:25, 20), ]
df_portfolio_20$symbol <- as.factor(df_portfolio_20$symbol)
df_portfolio_20$symbol <- as.integer(df_portfolio_20$symbol)
df_portfolio_25 <- df_portfolio

alpha = 0.3
UCB_portfolio_5 <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_10 <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_15 <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_20 <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_25 <- LinUCBDisjointPolicy$new(alpha = alpha)

nu = 0.3
TS_portfolio_5 <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_10 <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_15 <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_20 <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_25 <- LinUCBDisjointPolicy$new(nu)

bandit_portfolio_5 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio_5, randomize = FALSE, replacement = FALSE)
bandit_portfolio_10 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio_10, randomize = FALSE, replacement = FALSE)
bandit_portfolio_15 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio_15, randomize = FALSE, replacement = FALSE)
bandit_portfolio_20 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio_20, randomize = FALSE, replacement = FALSE)
bandit_portfolio_25 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol, data = df_portfolio_25, randomize = FALSE, replacement = FALSE)

agent_portfolio_UCB_5 <- Agent$new(policy = UCB_portfolio_5, bandit = bandit_portfolio_5, name = "UCB a=0.3 k=5")
agent_portfolio_UCB_10 <- Agent$new(policy = UCB_portfolio_10, bandit = bandit_portfolio_10, name = "UCB a=0.3 k=10")
agent_portfolio_UCB_15 <- Agent$new(policy = UCB_portfolio_15, bandit = bandit_portfolio_15, name = "UCB a=0.3 k=15")
agent_portfolio_UCB_20 <- Agent$new(policy = UCB_portfolio_20, bandit = bandit_portfolio_20, name = "UCB a=0.3 k=20")
agent_portfolio_UCB_25 <- Agent$new(policy = UCB_portfolio_25, bandit = bandit_portfolio_25, name = "UCB a=0.3 k=25")

agent_portfolio_TS_5 <- Agent$new(policy = TS_portfolio_5, bandit = bandit_portfolio_5, name = "TS v=0.3 k=5")
agent_portfolio_TS_10 <- Agent$new(policy = TS_portfolio_10, bandit = bandit_portfolio_10, name = "TS v=0.3 k=10")
agent_portfolio_TS_15 <- Agent$new(policy = TS_portfolio_15, bandit = bandit_portfolio_15, name = "TS v=0.3 k=15")
agent_portfolio_TS_20 <- Agent$new(policy = TS_portfolio_20, bandit = bandit_portfolio_20, name = "TS v=0.3 k=20")
agent_portfolio_TS_25 <- Agent$new(policy = TS_portfolio_25, bandit = bandit_portfolio_25, name = "TS v=0.3 k=25")

size_sim=100000
n_sim=14

# simulator UCB
run_simulator(list(agent_portfolio_UCB_5, agent_portfolio_UCB_10, agent_portfolio_UCB_15,
                   agent_portfolio_UCB_20, agent_portfolio_UCB_25))

# simulator TS
run_simulator(list(agent_portfolio_TS_5, agent_portfolio_TS_10, agent_portfolio_TS_15,
                   agent_portfolio_TS_20, agent_portfolio_TS_25))

alpha=0.1
UCB_portfolio_5_context <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_10_context <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_15_context <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_20_context <- LinUCBDisjointPolicy$new(alpha = alpha)
UCB_portfolio_25_context <- LinUCBDisjointPolicy$new(alpha = alpha)

nu = 0.01
TS_portfolio_5_context <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_10_context <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_15_context <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_20_context <- LinUCBDisjointPolicy$new(nu)
TS_portfolio_25_context <- LinUCBDisjointPolicy$new(nu)

bandit_portfolio_context_5 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio_5, randomize = FALSE, replacement = FALSE)
bandit_portfolio_context_10 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio_10, randomize = FALSE, replacement = FALSE)
bandit_portfolio_context_15 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio_15, randomize = FALSE, replacement = FALSE)
bandit_portfolio_context_20 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio_20, randomize = FALSE, replacement = FALSE)
bandit_portfolio_context_25 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ symbol| return + RSI + implied_vol, data = df_portfolio_25, randomize = FALSE, replacement = FALSE)

agent_portfolio_UCB_context_5 <- Agent$new(policy = UCB_portfolio_5_context, bandit = bandit_portfolio_context_5, name = "CUCB a=0.1 k=5")
agent_portfolio_UCB_context_10 <- Agent$new(policy = UCB_portfolio_10_context, bandit = bandit_portfolio_context_10, name = "CUCB a=0.1 k=10")
agent_portfolio_UCB_context_15 <- Agent$new(policy = UCB_portfolio_15_context, bandit = bandit_portfolio_context_15, name = "CUCB a=0.1 k=15")
agent_portfolio_UCB_context_20 <- Agent$new(policy = UCB_portfolio_20_context, bandit = bandit_portfolio_context_20, name = "CUCB a=0.1 k=20")
agent_portfolio_UCB_context_25 <- Agent$new(policy = UCB_portfolio_25_context, bandit = bandit_portfolio_context_25, name = "CUCB a=0.1 k=25")

agent_portfolio_TS_context_5 <- Agent$new(policy = TS_portfolio_5_context, bandit = bandit_portfolio_context_5, name = "CTS v=0.01 k=5")
agent_portfolio_TS_context_10 <- Agent$new(policy = TS_portfolio_10_context, bandit = bandit_portfolio_context_10, name = "CTS v=0.01 k=10")
agent_portfolio_TS_context_15 <- Agent$new(policy = TS_portfolio_15_context, bandit = bandit_portfolio_context_15, name = "CTS v=0.01 k=15")
agent_portfolio_TS_context_20 <- Agent$new(policy = TS_portfolio_20_context, bandit = bandit_portfolio_context_20, name = "CTS v=0.01 k=20")
agent_portfolio_TS_context_25 <- Agent$new(policy = TS_portfolio_25_context, bandit = bandit_portfolio_context_25, name = "CTS v=0.01 k=25")

# simulator UCB context
run_simulator(list(agent_portfolio_UCB_context_5, agent_portfolio_UCB_context_10, agent_portfolio_UCB_context_15,
                   agent_portfolio_UCB_context_20, agent_portfolio_UCB_context_25))

# simulator TS context
run_simulator(list(agent_portfolio_TS_context_5, agent_portfolio_TS_context_10, agent_portfolio_TS_context_15,
                   agent_portfolio_TS_context_20, agent_portfolio_TS_context_25))





