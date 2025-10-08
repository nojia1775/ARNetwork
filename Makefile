CXX = g++

CXXFLAGS = -std=c++2a -Wall -Wextra -Werror -g -MMD

OBJS_DIR = obj

SRCS_TRAIN = train.cpp

SRCS = compute.cpp

OBJS_TRAIN = $(SRCS_TRAIN:%.cpp=$(OBJS_DIR)/%.o)

OBJS = $(SRCS:%.cpp=$(OBJS_DIR)/%.o)

DEPS = $(OBJS:.o=.d)

DEPS_TRAIN = $(OBJS_TRAIN:.o=.d)

NAME = compute

NAME_TRAIN = train

all: train compute

$(NAME): $(OBJS)
	make -C ARNetwork
	$(CXX) $(CXXFLAGS) $^ ARNetwork/arnetwork.a -o $@

$(NAME_TRAIN): $(OBJS_TRAIN)
	make -C ARNetwork
	$(CXX) $(CXXFLAGS) $^ ARNetwork/arnetwork.a -o $@
	
$(OBJS_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	make clean -C ARNetwork
	rm -rf $(OBJS_DIR) $(DEPS) $(DEPS_TRAIN)

fclean: clean
	make fclean -C ARNetwork
	rm -f $(NAME) $(NAME_TRAIN)

re: fclean all

-include $(DEPS)
-include $(DEPS_TRAIN)

.PHONY: all clean fclean re