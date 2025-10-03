#include "../include/PairFunction.hpp"

// sigmoid / ReLU / leakyReLU / identity / BCE / MSE
PairFunction::PairFunction(const std::string& function) : _act(nullptr), _dact(nullptr), _loss(nullptr), _dloss(nullptr)
{
	if (function == "sigmoid")
	{
		_act = sigmoid;
		_dact = derived_sigmoid;
	}
	else if (function == "ReLU")
	{
		_act = ReLU;
		_dact = derived_ReLU;
	}
	else if (function == "leakyReLU")
	{
		_act = leakyReLU;
		_dact = derived_leakyReLU;
	}
	else if (function == "identity")
	{
		_act = identity;
		_dact = derived_identity;
	}
	else if (function == "BCE")
	{
		_loss = BCE;
		_dloss = derived_BCE;
	}
	else if (function == "MSE")
	{
		_loss = MSE;
		_dloss = derived_MSE;
	}
	else
		throw Error("Error: unknown function");
}