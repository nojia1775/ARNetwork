#pragma once

#include "../../linear_algebra/include/LinearAlgebra.hpp"
#include <random>
#include <cmath>
#include <algorithm>

class	ARNetwork
{
	private:
		Vector<double>				_inputs;
		Vector<double>				_outputs;
		std::vector<Matrix<double>>		_weights;
		std::vector<std::vector<double>>	_bias;
		double					_learning_rate;

	public:
							ARNetwork(const size_t& inputs, const size_t& hidden_layers, const size_t& hidden_neurals, const size_t& outputs);
							~ARNetwork(void) {}

							ARNetwork(const ARNetwork& arn);
		ARNetwork				operator=(const ARNetwork& arn);

		const Vector<double>&			get_inputs(void) const { return _inputs; }
		const double&				get_input(const size_t& index) { if (index > _inputs.dimension() - 1) throw Error("Error: out of range"); else return _inputs[index]; }
		const std::vector<Matrix<double>>&	get_weights(void) const { return _weights; }
		const Matrix<double>&			get_weights(const size_t& layer) const { if (layer > _weights.size() - 1) throw Error("Error: index out of range"); else return _weights[layer]; }
		const std::vector<std::vector<double>>&	get_bias(void) const { return _bias; }
		const std::vector<double>&		get_bias(const size_t& index) const { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else return _bias[index]; }
		const double&				get_bias(const size_t& i, const size_t& j) const;
		const double&				get_learning_rate(void) const { return _learning_rate; }
		const Vector<double>&			get_outputs(void) const { return _outputs; }
		const double&				get_output(const size_t& index) { if (index > _outputs.dimension() - 1) throw Error("Error: index out of range"); else return _outputs[index]; }

		size_t					nbr_inputs(void) const { return _inputs.dimension(); }
		size_t					nbr_hidden_layers(void) const { return _weights.size() - 1; }
		size_t					nbr_hidden_neurals(void) const { return _weights.size() == 0 ? 0 : _weights[0].getNbrLines(); }
		size_t					nbr_bias(void) const { return _bias.size(); }
		size_t					nbr_outputs(void) const { return _outputs.dimension(); }

		void					set_inputs(const Vector<double>& inputs) { _inputs = inputs; }
		void					set_weights(std::vector<Matrix<double>>& weights) { _weights = weights; }
		void					set_weights(const size_t& index, const Matrix<double>& weights) { if (index > _weights.size() - 1) throw Error("Error: index out of range"); else _weights[index] = weights; }
		void					set_bias(const std::vector<std::vector<double>>& bias) { _bias = bias; }
		void					set_bias(const size_t& index, const std::vector<double>& bias) { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else _bias[index] = bias; }
		void					set_bias(const size_t& i, const size_t& j, const double& bias);
		void					set_learning_rate(const double& learning_rate) { _learning_rate = learning_rate; }

		Vector<double>				feed_forward(const Vector<double>& inputs, double (*layer_activation)(const double&), double (*output_activation)(const double&));
		double					back_propagation(double (*loss)(const double&, const double&), double (*d_loss)(const double&, const double&), double (*d_layer_activation)(const double&), double (*d_output_activation)(const double&), const std::vector<double>& y);
};

inline std::mt19937&	global_urng(void)
{
	std::random_device rd;
	static std::mt19937 gen(rd());
	return gen;
}

inline double	random_double(const double& min, const double& max)
{
	std::uniform_real_distribution<double> dist(min, max);
	return dist(global_urng());
}

// activation functions

inline double	ReLU(const double& x) { return x <= 0 ? 0 : x; }
inline double	derivatedReLU(const double& x) { return x <= 0 ? 0 : 1; }

inline double	leakyReLU(const double& x) { return x <= 0 ? x * 0.01 : x; }
inline double	derivatedLeakyReLU(const double& x) { return x <= 0 ? 0.01 : 1; }

inline double	sigmoid(const double& x) { return 1 / (1 + exp(-x)); }
inline double	derivatedSigmoid(const double& x) { return sigmoid(x) * (1 - sigmoid(x)); }

inline double	identity(const double& x) { return x; }
inline double	derivatedIdentity(const double& x) { return x / std::abs(x); }

inline double	tanH(const double& x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
inline double	derivatedTanH(const double& x) { return 1 - pow(tanH(x), 2); }

std::vector<double>	softMax(const std::vector<double>& input);
std::vector<double>	derivatedSoftMax(const std::vector<double>& inputs);

// loss functions

inline double	BCE(const double& yPred, const double& yTrue) { return - yTrue * log(yPred) - (1 - yTrue) * log(1 - yPred); }
inline double	derivatedBCE(const double& yPred, const double& yTrue) { return - yTrue / (yPred + 1e-7) + (1 - yTrue) / (1 - yPred + 1e-7); }

inline double	MSE(const double& yPred, const double& yTrue) { return pow(yPred - yTrue, 2) / 2; }
inline double	derivatedMSE(const double& yPred, const double& yTrue) { return yPred - yTrue; }

double	crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);
double	derivatedCrossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue);