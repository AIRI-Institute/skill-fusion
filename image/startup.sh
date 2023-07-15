#!/bin/bash

# config the screen resolution

#export WANDB_NAME=LUNAR_sweep_$vnc_port+_$jup_port

sed -i "s/1920x1080/1920x1080/" /usr/local/bin/xvfb.sh

# some configurations for LXDE
mkdir -p /root/.config/pcmanfm/LXDE/
ln -sf /usr/local/share/doro-lxde-wallpapers/desktop-items-0.conf /root/.config/pcmanfm/LXDE/


# start all the services
exec /bin/tini -- /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf

exec /bin/bash -jupyter notebook --allow-root
 