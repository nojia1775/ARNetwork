#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include "../../linear_algebra/include/Error.hpp"

// activation functions

inline double	ReLU(const double& x) { return x <= 0 ? 0 : x; }
inline double	derived_ReLU(const double& x) { return x <= 0 ? 0 : 1; }

inline double	leakyReLU(const double& x) { return x <= 0 ? x * 0.01 : x; }
inline double	derived_leakyReLU(const double& x) { return x <= 0 ? 0.01 : 1; }

inline double	sigmoid(const double& x) { return 1 / (1 + exp(-x)); }
inline double	derived_sigmoid(const double& x) { return sigmoid(x) * (1 - sigmoid(x)); }

inline double	identity(const double& x) { return x; }
inline double	derived_identity(const double& x) { return x / std::abs(x); }

inline double	tanH(const double& x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
inline double	derived_tanH(const double& x) { return 1 - pow(tanH(x), 2); }

std::vector<double>	softMax(const std::vector<double>& input);
std::vector<double>	derived_softMax(const std::vector<double>& inputs);

// loss functions

inline double	BCE(const double& yPred, const double& yTrue) { return - yTrue * log(yPred) - (1 - yTrue) * log(1 - yPred); }
inline double	derived_BCE(const double& yPred, const double& yTrue) { return - yTrue / (yPred + 1e-7) + (1 - yTrue) / (1 - yPred + 1e-7); }

inline double	MSE(const double& yPred, const double& yTrue) { return pow(yPred - yTrue, 2) / 2; }
inline double	derived_MSE(const double& yPred, const double& yTrue) { return yPred - yTrue; }

double	crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);
double	derived_crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);

class	PairFunction
{
	typedef double (*activation_function)(const double&);
	typedef double (*loss_function)(const double&, const double&);

	private:
		double			(*_act)(const double&);
		double			(*_dact)(const double&);
		double			(*_loss)(const double&, const double&);
		double			(*_dloss)(const double&, const double&);

	public:
					PairFunction(const std::string& function);
					~PairFunction(void) {}

					PairFunction(double (*f)(const double&), double (*df)(const double&)) : _act(f), _dact(df), _loss(nullptr), _dloss(nullptr) {}
					PairFunction(double (*loss)(const double&, const double&), double (*dloss)(const double&, const double&)) : _act(nullptr), _dact(nullptr), _loss(loss), _dloss(dloss) {}

		activation_function	get_activation_function(void) const { return _act; }
		activation_function	get_derived_activation_function(void) const { return _dact; }
		loss_function		get_loss_function(void) const { return _loss; }
		loss_function		get_derived_loss_function(void) const { return _dloss; }

		double			foo(const double& x) const { if (!_act) throw Error("Error: there is no activation function"); return _act(x); }
		double			foo(const double& a, const double& b) const { if (!_loss) throw Error("Error: there is no loss function"); return _loss(a, b); }
		double			derived_foo(const double& x) const { if (!_dact) throw Error("Error: there is no derived activation function"); return _dact(x); }
		double			derived_foo(const double& a, const double& b) const { if (!_dloss) throw Error("Error: there is no derived loss function"); return _dloss(a, b); }
};