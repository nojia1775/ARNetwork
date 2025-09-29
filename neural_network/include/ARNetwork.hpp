#pragma once

#include "../../linear_algebra/include/LinearAlgebra.hpp"

class	ARNetwork
{
	private:
		Vector<double>				_inputs;
		Vector<double>				_outputs;
		std::vector<Matrix<double>>		_weights;
		std::vector<double>			_bias;
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
		const std::vector<double>&		get_bias(void) const { return _bias; }
		const double&				get_bias(const size_t& index) const { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else return _bias[index]; }
		const double&				get_learning_rate(void) const { return _learning_rate; }
		const Vector<double>&			get_outputs(void) const { return _outputs; }
		const double&				get_output(const size_t& index) { if (index > _outputs.dimension() - 1) throw Error("Error: index out of range"); else return _outputs[index]; }

		size_t					nbr_inputs(void) const { return _inputs.dimension(); }
		size_t					nbr_hidden_layers(void) const { return _weights.size(); }
		size_t					nbr_hidden_neurals(void) const { return _weights.size() == 0 ? 0 : _weights[0].getNbrLines(); }
		size_t					nbr_bias(void) const { return _bias.size(); }
		size_t					nbr_outputs(void) const { return _outputs.dimension(); }

		void					set_inputs(const Vector<double>& inputs) { _inputs = inputs; }
		void					set_weights(std::vector<Matrix<double>>& weights) { _weights = weights; }
		void					set_weights(const size_t& index, const Matrix<double>& weights) { if (index > _weights.size() - 1) throw Error("Error: index out of range"); else _weights[index] = weights; }
		void					set_bias(const std::vector<double>& bias) { _bias = bias; }
		void					set_bias(const size_t& index, const double& bias) { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else _bias[index] = bias; }
		void					set_learning_rate(const double& learning_rate) { _learning_rate = learning_rate; }
};