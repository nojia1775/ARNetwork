#include "../include/ARNetwork.hpp"

std::vector<double>	softMax(const std::vector<double>& input)
{
	std::vector<double> output(input.size());
	double maxVal = *std::max_element(input.begin(), input.end());
	double sumExp = 0.0;
	for (size_t i = 0; i < input.size(); ++i)
	{
		output[i] = std::exp(input[i] - maxVal);
		sumExp += output[i];
	}
	for (size_t i = 0; i < output.size(); ++i)
		output[i] /= sumExp;
	return output;
}

std::vector<double>	derivatedSoftMax(const std::vector<double>& inputs)
{
	std::vector<double> result(inputs.size());

	for (size_t i = 0 ; i < inputs.size() ; i++)
	{
		double sum = 0;
		for (size_t j = 0 ; j < inputs.size() ; j++)
			sum += exp(inputs[j]);
		result[i] = exp(inputs[i]) / sum;
	}
	return result;
}

double	crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue)
{
	double loss = 0.0f;
	for (size_t i = 0; i < yTrue.size(); ++i)
		loss -= std::log(yPred[i] + std::numeric_limits<double>::epsilon());
	return loss;
}

double	derivatedCrossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue)
{
	double loss = 0;
	for (size_t i = 0 ; i < yPred.size() ; i++)
		loss += yTrue[i] * std::log(yPred[i] + std::numeric_limits<double>::epsilon());
	return -loss;
}