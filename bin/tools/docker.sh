#!/bin/bash
docker-compose up -d
docker-compose exec $(docker-compose ps --service) bash
