# scenarios folder

This folder only configures the supply chain scenario used in Chapter 3 and Chapter 4.
It keeps input and output simple and direct.
Everything is plain Python.

Files

- scenario.py
  Defines the scenario object and the parameters.
  It also provides helper functions used later by the problem module and the RL environment
  ramp-up and effective capacity
  tariff-inclusive delivered cost
  demand construction (potential demand, realised demand, and demand loss)
  relationship premium on material cost for the candidate plant

  The key function is validate_assumptions.
  It checks the must-have story rules and Chapter 3.1 assumptions.
  If you break a rule it raises a clear error message.

- story_case.py
  Builds one numeric scenario that follows the story.
  You can edit numbers here.
  The file calls validate_assumptions before returning the scenario.

How it matches the assumptions

- three tiers (suppliers, plants, markets) are explicit in the scenario object
- four outbound edges are represented by c_out and tau dictionaries
- tariff uncertainty is represented by a Markov chain with P and regimes
- initial escalation is enforced by xi_2_forced and validated using tau(M1,D1,xi2) > tau(M1,D1,xi1)
- demand construction follows Chapter 3, including demand loss
- make-to-order and no inventory is respected because the scenario has no inventory state
- activation lag is fixed to 1 and checked
- ramp-up uses rho(age)=1-exp(-kappa*age) for the candidate plant
- supplier capacity W_bar and inbound costs c_in are included
- relationship premium is applied to the candidate plant for the first H_rel periods
- qualification and deposit cost window (qual_ell) and cost (qual_G) are stored in reconfig
- salvage credit S_salv is stored in reconfig for full relocation

Quick use

from scenarios.story_case import build_story_case
scenario = build_story_case()