FROM nginx:1.27-alpine

# Create a named user/group with uid/gid=1000 so nginx `user` directive can resolve it.
RUN addgroup -g 1000 -S xima && adduser -u 1000 -S -G xima xima

# Ensure nginx can write runtime files when running as non-root.
RUN mkdir -p /var/cache/nginx /var/run /var/log/nginx && \
    chown -R 1000:1000 /var/cache/nginx /var/run /var/log/nginx

# Proxy cache dir (used for auth micro-cache).
RUN mkdir -p /var/cache/nginx/auth && \
    chown -R 1000:1000 /var/cache/nginx/auth

# Auth mode selectable configs (default: open)
RUN mkdir -p /etc/nginx/xima && \
    chown -R 1000:1000 /etc/nginx

COPY --chown=1000:1000 nginx.conf /etc/nginx/xima/nginx.conf
COPY --chown=1000:1000 nginx.open.conf /etc/nginx/xima/nginx.open.conf
COPY --chown=1000:1000 docker-entrypoint.d/40-xima-auth-mode.sh /docker-entrypoint.d/40-xima-auth-mode.sh

RUN chmod +x /docker-entrypoint.d/40-xima-auth-mode.sh && \
    cp /etc/nginx/xima/nginx.open.conf /etc/nginx/nginx.conf

EXPOSE 8080
