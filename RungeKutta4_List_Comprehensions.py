class RungeKutta4(ODESolver):
    def advance(self):
        u = self.u[self.n]
        t = self.t[self.n]
        dt = self.t[self.n + 1] - t
        dt2 = dt / 2.0

        k1 = self.f(u, t)
        k2 = self.f(u + dt2 * k1, t + dt2)
        k3 = self.f(u + dt2 * k2, t + dt2)
        k4 = self.f(u + dt * k3, t + dt)

        increments = [k1, k2, k2, k3, k3, k4]
        weighted_sum = sum([(1/6.0) * dt * increment for increment in increments])

        u_new = u + weighted_sum
        return u_new
