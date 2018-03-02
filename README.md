# This is agent F.

Policy gradient with very basic environment and "deep" neural network.
The agent will decide, on each timestep, of an asset portfolio weight vector and then proceed to the transactions every day. 
The optimization is done with respect to the differential Sharpe Ratio.
TODO:
- Include some temporality and recurrence in the model or in the data processing.
- Complexify the reward to give more accurate credit to the right actions.
