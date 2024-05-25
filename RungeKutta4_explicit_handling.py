class RungeKutta4(ODESolver):
    def advance(self):
        u_n = self.u[self.n]
        t_n = self.t[self.n]
        dt = self.t[self.n + 1] - t_n
        dt2 = dt / 2.0

        k1 = self.f(u_n, t_n)
        k2 = self.f(u_n + dt2 * k1, t_n + dt2)
        k3 = self.f(u_n + dt2 * k2, t_n + dt2)
        k4 = self.f(u_n + dt * k3, t_n + dt)

        u_new = u_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u_new
