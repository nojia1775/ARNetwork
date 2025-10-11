#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include "../../linear_algebra/include/LinearAlgebra.hpp"

// activation functions

inline double	ReLU(const double& x) { return x <= 0 ? 0 : x; }
inline double	derived_ReLU(const double& x) { return x <= 0 ? 0 : 1; }

inline double	leakyReLU(const double& x) { return x <= 0 ? x * 0.01 : x; }
inline double	derived_leakyReLU(const double& x) { return x <= 0 ? 0.01 : 1; }

inline double	tanh(const double& x) { return std::tanh(x); }
inline double	derived_tanh(const double& x) { return 1 - std::tanh(x) * std::tanh(x); }

inline double	sigmoid(const double& x) { return 1 / (1 + exp(-x)); }
inline double	derived_sigmoid(const double& x) { return sigmoid(x) * (1 - sigmoid(x)); }

inline double	identity(const double& x) { return x; }
inline double	derived_identity(const double& x) { return x >= 0 ? 1 : -1; }

inline double	tanH(const double& x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
inline double	derived_tanH(const double& x) { return 1 - pow(tanH(x), 2); }

std::vector<double>	softMax(const std::vector<double>& input);
std::vector<double>	derived_softMax(const std::vector<double>& inputs);

// loss functions

double	BCE(const Vector<double>& y_pred, const Vector<double>& y);
Vector<double>	derived_BCE(const Vector<double>& y_pred, const Vector<double>& y);

double	MSE(const Vector<double>& y_pred, const Vector<double>& y);
Vector<double>	derived_MSE(const Vector<double>& y_pred, const Vector<double>& y);

double	crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);
double	derived_crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);

class	PairFunction
{
	typedef double (*activation_function)(const double&);
	typedef double (*loss_function)(const Vector<double>&, const Vector<double>&);
	typedef Vector<double> (*derived_loss_function)(const Vector<double>&, const Vector<double>&);

	private:
		double			(*_act)(const double&);
		double			(*_dact)(const double&);
		double			(*_loss)(const Vector<double>&, const Vector<double>&);
		Vector<double>		(*_dloss)(const Vector<double>&, const Vector<double>&);
		std::string		_activation_name;
		std::string		_loss_name;

	public:
					PairFunction(const std::string& function);
					~PairFunction(void) {}

					PairFunction(double (*f)(const double&), double (*df)(const double&), const std::string& activation_name) : _act(f), _dact(df), _loss(nullptr), _dloss(nullptr), _activation_name(activation_name), _loss_name() {}
					PairFunction(double (*loss)(const Vector<double>&, const Vector<double>&), Vector<double> (*dloss)(const Vector<double>&, const Vector<double>&), const std::string& loss_name) : _act(nullptr), _dact(nullptr), _loss(loss), _dloss(dloss), _activation_name(), _loss_name(loss_name) {}

		activation_function	get_activation_function(void) const { return _act; }
		activation_function	get_derived_activation_function(void) const { return _dact; }
		loss_function		get_loss_function(void) const { return _loss; }
		derived_loss_function	get_derived_loss_function(void) const { return _dloss; }
		const std::string&	get_activation_name(void) const { return _activation_name; }
		const std::string&	get_loss_name(void) const { return _loss_name; }

		double			foo(const double& x) const { if (!_act) throw Error("Error: there is no activation function"); return _act(x); }
		double			foo(const Vector<double>& a, const Vector<double>& b) const { if (!_loss) throw Error("Error: there is no loss function"); return _loss(a, b); }
		double			derived_foo(const double& x) const { if (!_dact) throw Error("Error: there is no derived activation function"); return _dact(x); }
		Vector<double>		derived_foo(const Vector<double>& a, const Vector<double>& b) const { if (!_dloss) throw Error("Error: there is no derived loss function"); return _dloss(a, b); }
};