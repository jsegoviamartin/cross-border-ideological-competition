class RungeKutta4(ODESolver):
    def advance(self):
        u = self.u
        f = self.f
        n = self.n
        t = self.t
        dt = t[n + 1] - t[n]

        k1 = f(u[n], t[n])
        k2 = f(u[n] + 0.5 * dt * k1, t[n] + 0.5 * dt)
        k3 = f(u[n] + 0.5 * dt * k2, t[n] + 0.5 * dt)
        k4 = f(u[n] + dt * k3, t[n] + dt)

        u[n + 1] = u[n] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u[n + 1]
