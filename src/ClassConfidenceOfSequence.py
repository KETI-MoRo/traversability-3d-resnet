# [ Reference ]
#
# Kim, Kwanghyun, et al.
# "Probabilistic ship detection and classification using deep learning."
# Applied Sciences 8.6 (2018): 936.
# https://www.mdpi.com/2076-3417/8/6/936


class ClassConfidence():
    def __init__(self,class_num_):

        # Number of class
        self.C = class_num_
     
        # (2023-09-03) Introduce epsilon to prevent ZeroDivisionError
        #self.epsilon = pow( 10, -300 )
        self.epsilon = 0.001

        # the class confidence of a sequnce
        # Eq. 14
        self.f = []

        # Algorithm 1 > Step 1 > line 3
        self.f_0 = []
        for i in range(self.C):
            self.f_0.append(1/self.C)


    # Initialize
    def reset(self):

        # the class confidence of a sequnce
        # Eq. 14
        self.f = []


    def evaluate(self, p_t):
        """
        Algorithm 1 > Step 3 :
        Evaluate the class confidence of a sequence recursively.

        :param p_t: The detection confidence for the image at time t.
        :return: The class confidence of a sequence at time t.
        """
        
        # Check the length of the input detection confidence.
        assert self.C == len(p_t), 'The len of detection confidence should be equal to the class num.'

        # Algorithm 1 > Step 1 > line 2
        # the class confidence of a sequnce
        # Eq. 14
        if len(self.f) == 0:

            # If p_t is the first frame of the sequence,
            # the p_t will directly set to f_t.
            self.f.append(p_t)

            return self.get_confidence()
        
        ######################
        # Algorithm 1 > Step 3

        # Get f^{t-1}
        f_t_old = self.get_confidence()

        rho_t = []
        f_t = []

        for k in range(self.C):

            # Algorithm 1 > Step 3 > line 1

            rho = 0
            for k_prime in range(self.C):

                # (2023-09-03) Introduce epsilon to prevent ZeroDivisionError
                if False :  # [Retained for reference] The original implementation
                    temp = (p_t[k_prime] * f_t_old[k_prime] * self.f_0[k]) \
                        / (p_t[k] * f_t_old[k] * self.f_0[k_prime])
                    rho = rho + temp

                else :  # Apply epsilon
                    temp = (p_t[k_prime] * f_t_old[k_prime] * self.f_0[k]) \
                        / (p_t[k] * f_t_old[k] * self.f_0[k_prime] + self.epsilon)
                    rho = rho + temp

            rho_t.append(rho)

            # Algorithm 1 > Step 3 > line 2
            f_t.append(1/rho)
        
        # Append f_t to the class attribute list `f`.
        self.f.append(f_t)

        return f_t


    # Return the most recent class confidence
    def get_confidence(self):
        return(self.f[len(self.f)-1])
