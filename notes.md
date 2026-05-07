# Development Notes

Keep a list of things to bear in mind, design choices:

* At the moment, we are using model.DEFAULT_PARAMS to count the number of parameters in a simple way, which is needed for applying the complexity penalty. But if it is not defined then infinite loss is assigned in scoring.scoring._score_one(...)! So we need to guarantee model.DEFAULT_PARAMS is appropriately defined. 
* The `TranslationSchema` assumes that the jax code is only the translated model and parameter estimator. Does this work with the fact that we also need model.DEFAULT_PARAMS to be defined?