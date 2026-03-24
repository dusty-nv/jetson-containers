FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN sed -i 's/^user  nginx;/# user  nginx;/' /etc/nginx/nginx.conf && \
    sed -i 's|/run/nginx.pid|/tmp/nginx.pid|' /etc/nginx/nginx.conf && \
    chmod -R 777 /var/cache/nginx /var/log/nginx

EXPOSE 8080
