# This is agent F.

Policy gradient with very basic environment and "deep" neural network.
The agent will decide, on each timestep, of an asset portfolio weight vector and then proceed to the transactions every day. 
The optimization is done with respect to the differential Sharpe Ratio.

Warning: This is a very rough experimental toy project.

TODO:
- Include a better form of normalization
- Include a way to assess efficiency of the learning process
- Improve loss function
- Include some temporality and recurrence in the model or in the data processing.
- Complexify the reward to give more accurate credit to the right actions.
