#include "../include/PairFunction.hpp"

// sigmoid / ReLU / leakyReLU / identity / BCE / MSE
PairFunction::PairFunction(const std::string& function) : _act(nullptr), _dact(nullptr), _loss(nullptr), _dloss(nullptr)
{
	if (function == "sigmoid")
	{
		_activation_name = function;
		_act = sigmoid;
		_dact = derived_sigmoid;
	}
	else if (function == "ReLU")
	{
		_activation_name = function;
		_act = ReLU;
		_dact = derived_ReLU;
	}
	else if (function == "leakyReLU")
	{
		_activation_name = function;
		_act = leakyReLU;
		_dact = derived_leakyReLU;
	}
	else if (function == "identity")
	{
		_activation_name = function;
		_act = identity;
		_dact = derived_identity;
	}
	else if (function == "BCE")
	{
		_loss_name = function;
		_loss = BCE;
		_dloss = derived_BCE;
	}
	else if (function == "MSE")
	{
		_loss_name = function;
		_loss = MSE;
		_dloss = derived_MSE;
	}
	else if (function == "tanh")
	{
		_activation_name = function;
		_act = tanh;
		_dact = derived_tanh;
	}
	else
		throw Error("Error: unknown function");
}