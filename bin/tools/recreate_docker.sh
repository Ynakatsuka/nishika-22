#!/bin/bash
docker-compose up -d --force-recreate
docker-compose exec $(docker-compose ps --service) bash
