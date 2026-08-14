-- FACTORY GIVES (experimental, simple-avatar fork) — 14 Aug 2026
-- Sellers vote the Factory's giving percentage + cause; charities apply for help.
-- Safe to re-run.

create table if not exists charity_votes (
  auth_id uuid primary key,
  pct int not null check (pct in (1,2,3)),
  charity text not null,
  created_at timestamptz default now()
);
alter table charity_votes enable row level security;
drop policy if exists "vote own" on charity_votes;
create policy "vote own" on charity_votes
  for all using (auth.uid() = auth_id) with check (auth.uid() = auth_id);
drop policy if exists "founder reads votes" on charity_votes;
create policy "founder reads votes" on charity_votes
  for select using (exists (select 1 from users u where u.auth_id = auth.uid() and u.is_founder));

create table if not exists charity_applications (
  id bigint generated always as identity primary key,
  name text not null,
  website text,
  reason text not null,
  created_at timestamptz default now()
);
alter table charity_applications enable row level security;
drop policy if exists "anyone applies" on charity_applications;
create policy "anyone applies" on charity_applications
  for insert with check (true);
drop policy if exists "founder reads applications" on charity_applications;
create policy "founder reads applications" on charity_applications
  for select using (exists (select 1 from users u where u.auth_id = auth.uid() and u.is_founder));
